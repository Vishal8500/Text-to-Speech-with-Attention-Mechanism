#!/usr/bin/env/python3
"""
Recipe for "direct" (speech -> semantics) SLU with wav2vec2.0_based transfer learning.

We encode input waveforms into features using a wav2vec2.0 model pretrained on ASR from HuggingFace (facebook/wav2vec2-base-960h),
then feed the features into a seq2seq model to map them to semantics.

(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Plantinga.)

Run using:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml

Authors
 * Boumadane Abdelmoumene 2021
 * Heba Abdelwahab  2021
 * Lugosch Loren 2020
"""

import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

# Define training procedure


class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # wav2vec forward pass
        wav2vec2_out = self.modules.wav2vec2(wavs, wav_lens)
        # SLU forward pass
        encoder_out = self.hparams.slu_enc(wav2vec2_out)
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN and self.step % show_results_every != 0:
            return p_seq, wav_lens
        else:
            p_tokens, _, _, _ = self.hparams.beam_searcher(
                encoder_out, wav_lens
            )
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        if stage == sb.Stage.TRAIN and self.step % show_results_every != 0:
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (self.step % show_results_every == 0):
            # Decode token terms to words
            predicted_semantics = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            for i in range(len(target_semantics)):
                print(" ".join(predicted_semantics[i]))
                print(" ".join(target_semantics[i]))
                print("")

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["SER"] = self.wer_metric.summarize("SER")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["SER"])
            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["SER"])

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "wave2vec2_lr": old_lr_wav2vec2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"SER": stage_stats["SER"]},
                min_keys=["SER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "wav2vec2_optimizer": self.wav2vec2_optimizer,
            "model_optimizer": self.optimizer,
        }


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # If we are testing on all the real data, including dev-real,
    # we shouldn't use dev-real as the validation set.
    if hparams["test_on_all_real"]:
        valid_path = hparams["csv_dev_synth"]
    else:
        valid_path = hparams["csv_dev_real"]
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_path,
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_real_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test_real"],
        replacements={"data_root": data_folder},
    )
    test_real_data = test_real_data.filtered_sorted(sort_key="duration")

    test_synth_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test_synth"],
        replacements={"data_root": data_folder},
    )
    test_synth_data = test_synth_data.filtered_sorted(sort_key="duration")

    all_real_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_all_real"],
        replacements={"data_root": data_folder},
    )
    all_real_data = all_real_data.filtered_sorted(sort_key="duration")

    datasets = [
        train_data,
        valid_data,
        test_real_data,
        test_synth_data,
        all_real_data,
    ]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return (
        train_data,
        valid_data,
        test_real_data,
        test_synth_data,
        all_real_data,
        tokenizer,
    )


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # plots results every N iterations

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing TAS)
    from prepare import prepare_TAS  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_TAS,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "type": "direct",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_set,
        valid_set,
        test_real_set,
        test_synth_set,
        all_real_set,
        tokenizer,
    ) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # Brain class initialization
    slu_brain = SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test (ALL real data)
    if slu_brain.hparams.test_on_all_real:
        slu_brain.hparams.test_wer_file = hparams["all_real_wer_file"]
        slu_brain.evaluate(
            all_real_set,
            test_loader_kwargs=hparams["dataloader_opts"],
            min_key="SER",
        )

    # Test (real data)
    slu_brain.hparams.test_wer_file = hparams["test_real_wer_file"]
    slu_brain.evaluate(
        test_real_set,
        test_loader_kwargs=hparams["dataloader_opts"],
        min_key="SER",
    )

    # Test (synth data)
    slu_brain.hparams.test_wer_file = hparams["test_synth_wer_file"]
    slu_brain.evaluate(
        test_synth_set,
        test_loader_kwargs=hparams["dataloader_opts"],
        min_key="SER",
    )
