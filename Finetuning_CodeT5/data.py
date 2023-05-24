"""
This file contains classes related to data manipulation:
generat stopping criteria and main trainer data collator.
"""

import random
import torch
import transformers
import datasets

from transformers import BatchEncoding
from transformers import StoppingCriteria


class StoppingCriteriaTokenIds(StoppingCriteria):
    def __init__(self, stop_ids, device="cuda"):
        super().__init__()
        self.stop_ids = torch.tensor(stop_ids).to(device)

    def __call__(self, input_ids, scores, *kwargs):
        # true if any alement is one of the stop ids in all rows
        return torch.all(
            torch.any(torch.isin(input_ids, self.stop_ids), dim=1, keepdim=False)
        ).item()


class DataCollatorNTP:
    def __init__(
        self,
        ds_data,
        tokenizer,
        min_encoder_seq_length,
        min_decoder_seq_length,
        encoder_seq_length,
        decoder_seq_length,
        append_special_token_to_input=None,
        add_max_padding=False,
    ):
        self.ds_data = ds_data
        self.tokenizer = tokenizer
        self.min_encoder_seq_length = min_encoder_seq_length
        self.min_decoder_seq_length = min_decoder_seq_length
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.append_special_token_to_input = append_special_token_to_input
        self.add_max_padding = add_max_padding

    def __call__(self, examples, return_tensors="pt"):
        """
        NOTE: the doc is from a standard codet5, where are changnes in the code

        This collate function is to be used with the CodeT5 model, which is a
        T5 model with a different tokenizer than the original T5 model.
        The tokenization is already done at this point and the examples are a list of
        tokenized inputs and labels. We cannot directly train on this data, though,
        because CodeT5 expects the inputs to be pre- and appended with special tokens,
        and that hasn't been done yet.
        The collate function first truncates the inputs and labels to the desired
        sequence lengths and then adds the special tokens. The decoder input is
        shifted to the right by one position as well.
        In particular, the following steps are performed after truncation:
        * The encoder input is prepended with the special token 1 (`<s>`) and
            appended with the special token 2 (`</s>`). Here's an example of how
            that is supposed to look like for CodeT5:
                >>> tokenizer = AutoTokenizer.from_pretrained("salesforce/codet5-base", use_fast=True, model_max_lenth=512)
                >>> tokenizer(text="foo", add_special_tokens=True).input_ids
                [1, 11351, 2]
                >>> tokenizer.decode([1, 11351, 2])
                '<s>foo</s>'
        * The labels are prepended with the special token 1 (`<s>`) and
            appended with the special token 2 (`</s>`). Again, here's an example
            of how that is supposed to look like for CodeT5:
                >>> tokenizer(text_target="foo", add_special_tokens=True).input_ids
                [1, 11351, 2]
        * For use as decoder inputs, the labels are then shifted to the right by one
            position and prepended with the special token 0 (`<pad>`). The special
            token 2 (`</s>`) is removed from the end of the labels. For the above
            example, this would look like this:
                >>> [0] + tokenizer(text_target="foo", add_special_tokens=True).input_ids[:-1]
                [0, 1, 11351]
        The collate function also adds padding to the inputs and labels. The padding
        token is 0 (`<pad>`) for inputs and -100 for labels. The padding is added to
        the end of the inputs and labels. The attention mask is padded as well.
        """

        if not isinstance(self.tokenizer, transformers.RobertaTokenizerFast):
            raise ValueError(
                "This collate function only works for CodeT5's RobertaTokenizerFast."
            )

        # the bos token is used to initialize the decoder_input_ids
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("The CodeT5 tokenizer should have a bos token.")

        # the eos token is used to terminate the input_ids, decoder_input_ids, and labels
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("The CodeT5 tokenizer should have an eos token.")

        # the pad token is used to initialize the decoder_input_ids
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The CodeT5 tokenizer should have a pad token.")

        encoder_seq_length = self.encoder_seq_length
        decoder_seq_length = self.decoder_seq_length

        do_labels_decoder_input = True

        if "target" in examples[0]:
            do_labels_decoder_input = False
            # format of data aready split by pivot point on str level
            encoder_ids_and_masks = self.tokenizer(
                [el["prior_context"] for el in examples], add_special_tokens=False
            )
            encoder_examples_ids = [
                [bos_token_id] + el for el in encoder_ids_and_masks["input_ids"]
            ]
            encoder_attention_mask = [
                [1] + el for el in encoder_ids_and_masks["attention_mask"]
            ]

            label_ids = [
                el + [eos_token_id]
                for el in self.tokenizer(
                    [el["target"] for el in examples],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            ]
            hole_id_ids = [
                el
                for el in self.tokenizer(
                    [el["id"] for el in examples],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            ]
            label_ids = [[len(l) + 1] + l + s for l, s in zip(label_ids, hole_id_ids)]
            max_label_length = max(len(label_seq) for label_seq in label_ids)
            label_ids_pt = torch.tensor(
                [
                    label_seq + [-100] * (max_label_length - len(label_seq))
                    for label_seq in label_ids
                ]
            )
        else:
            # format of pivot points dataset and data separatelly in ds_data
            examples_ids_and_masks = self.tokenizer(
                [
                    self.ds_data[example["split"]][example["data_idx"]]["content"]
                    for example in examples
                ]
            )
            pivotpoints = [example["pivot"] for example in examples]

            # either take pre-selected pivot point if exist
            # or randomly select a pivotpoint for each example
            # the seed is device specific and set by accelerate,
            # so this is deterministic. the midpoints will be
            # different each batch and epoch
            # NOTE: find how to set arbitrary seed for accelerate
            pivotpoints = [
                random.randint(
                    self.min_encoder_seq_length,
                    len(input_ids) - self.min_decoder_seq_length,
                )
                if pivot is None
                else pivot
                for input_ids, pivot in zip(
                    examples_ids_and_masks["input_ids"], pivotpoints
                )
            ]
            encoder_examples_ids = [
                example[: pivotpoints[i]]
                for i, example in enumerate(examples_ids_and_masks["input_ids"])
            ]
            encoder_attention_mask = [
                example[: pivotpoints[i]]
                for i, example in enumerate(examples_ids_and_masks["attention_mask"])
            ]
            label_ids = [
                example[pivotpoints[i] :]
                for i, example in enumerate(examples_ids_and_masks["input_ids"])
            ]

        if self.append_special_token_to_input is not None:
            append_special_token_to_input_id = self.tokenizer.vocab[
                self.append_special_token_to_input
            ]
            encoder_seq_length -= 1
            input_end = [append_special_token_to_input_id, eos_token_id]
            attention_mask_end = [1, 1]
        else:
            input_end = [eos_token_id]
            attention_mask_end = [1]

        # a column-wise representation of the examples is needed for the tokenizer.pad function
        example_dict: dict[str, list[list[int]]] = {}

        # truncate the input_ids to the encoder_seq_length from the beginning,
        # add the </s> (EOS) special token at the end, and
        # add the <s> (BOS) special token at the beginning if there is enough space
        def _add_bos_if_enough_space(
            label_seq: list[int], added_val: int = bos_token_id
        ) -> list[int]:
            if len(label_seq) < encoder_seq_length - 1 and (
                len(label_seq) == 0 or label_seq[0] != added_val
            ):
                return [added_val] + label_seq
            else:
                return label_seq

        example_dict["input_ids"] = [
            _add_bos_if_enough_space(example[-(encoder_seq_length - 1) :]) + input_end
            for example in encoder_examples_ids
        ]
        # truncate the attention_mask to the encoder_seq_length from the beginning
        example_dict["attention_mask"] = [
            _add_bos_if_enough_space(
                example[-(encoder_seq_length - 1) :],
                added_val=1,
            )
            + attention_mask_end
            for example in encoder_attention_mask
        ]

        if not do_labels_decoder_input:
            # temporarily add begin bos if not added,
            for i in range(len(example_dict["input_ids"])):
                if example_dict["input_ids"][i][0] != bos_token_id:
                    example_dict["input_ids"][i][0] = bos_token_id

        # add padding to the input_ids and attention_mask
        if not self.add_max_padding:
            encoding: BatchEncoding = self.tokenizer.pad(
                example_dict, return_tensors="pt"
            )
        else:
            encoding: BatchEncoding = self.tokenizer.pad(
                example_dict,
                padding="max_length",
                max_length=self.encoder_seq_length,
                return_tensors="pt",
            )

        if do_labels_decoder_input:
            # truncate the labels to the decoder_seq_length from the end,
            # add the <s> (BOS) special token at the beginning, and
            # add the </s> (EOS) special token at the end if there is enough space
            def _add_eos_if_enough_space(label_seq: list[int]) -> list[int]:
                if len(label_seq) < decoder_seq_length - 1 and (
                    len(label_seq) == 0 or label_seq[-1] != eos_token_id
                ):
                    return label_seq + [eos_token_id]
                else:
                    return label_seq

            labels = [
                [bos_token_id]
                + _add_eos_if_enough_space(label_seq=example[: decoder_seq_length - 1])
                for example in label_ids
            ]

            # add padding to the labels using -100 as the ignore_index
            if not self.add_max_padding:
                max_label_length = max(len(label_seq) for label_seq in labels)
            else:
                max_label_length = self.decoder_seq_length

            encoding["labels"] = torch.tensor(
                [
                    label_seq + [-100] * (max_label_length - len(label_seq))
                    for label_seq in labels
                ]
            )
            # shift the labels to the right by one position and
            # initialize the decoder_input_ids with the pad token
            # (we cannot leave this to `T5ForConditionalGeneration._shift_right` because the labels contain -100)
            decoder_input_ids = [
                [pad_token_id]
                + (
                    [bos_token_id]
                    + _add_eos_if_enough_space(example[: decoder_seq_length - 1])
                )[:-1]
                for example in label_ids
            ]

            if not self.add_max_padding:
                max_decoder_input_length = max(
                    len(decoder_input_id_seq)
                    for decoder_input_id_seq in decoder_input_ids
                )
            else:
                max_decoder_input_length = self.decoder_seq_length

            assert max_decoder_input_length == max_label_length
            encoding["decoder_input_ids"] = torch.tensor(
                [
                    decoder_input_id_seq
                    + [pad_token_id]
                    * (max_decoder_input_length - len(decoder_input_id_seq))
                    for decoder_input_id_seq in decoder_input_ids
                ]
            )
        else:
            encoding["labels"] = label_ids_pt
        return encoding
