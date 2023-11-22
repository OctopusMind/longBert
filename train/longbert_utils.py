import torch
from torch import Tensor


class LongBertPredictor:
    def __init__(self, model_name_or_path: str, encode_type='first_last_avg', **kwargs):
        from transformers import AutoModel, AutoTokenizer

        trust_remote_code = kwargs.pop('trust_remote_code', True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
        if hasattr(self.model, 'tokenizer'):
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        self.encode_type = encode_type

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def similarity(self, s1, s2, convert_to_numpy=True):
        emb1 = self.encode(s1)
        emb2 = self.encode(s2)
        sim_matrix = self.cos_sim(emb1, emb2)
        if convert_to_numpy:
            return sim_matrix.cpu().detach().numpy()
        return sim_matrix

    def encode(self, sentences, max_len=1024):
        inputs = self.tokenizer(
            sentences,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        return self.get_embedding(**inputs)

    def get_embedding(self, input_ids, attention_mask, token_type_ids=None):

        with torch.no_grad():
            model_output = self.model(
                input_ids, attention_mask, token_type_ids, output_hidden_states=True
            )

            if self.encode_type == 'first_last_avg':
                first = model_output.hidden_states[1]
                last = model_output.hidden_states[-1]
                seq_length = first.size(1)  # Sequence length

                first_avg = torch.avg_pool1d(
                    first.transpose(1, 2), kernel_size=seq_length
                ).squeeze(
                    -1
                )  # [batch, hid_size]
                last_avg = torch.avg_pool1d(
                    last.transpose(1, 2), kernel_size=seq_length
                ).squeeze(
                    -1
                )  # [batch, hid_size]
                return torch.avg_pool1d(
                    torch.cat(
                        [first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1
                    ).transpose(1, 2),
                    kernel_size=2,
                ).squeeze(-1)

            elif self.encode_type == 'last_avg':
                sequence_output = (
                    model_output.last_hidden_state
                )  # [batch_size, max_len, hidden_size]
                seq_length = sequence_output.size(1)
                return torch.avg_pool1d(
                    sequence_output.transpose(1, 2), kernel_size=seq_length
                ).squeeze(-1)

            elif self.encode_type == 'cls':
                sequence_output = model_output.last_hidden_state
                return sequence_output[:, 0]  # [batch, hid_size]

            elif self.encode_type == 'pooler':
                return model_output.pooler_output  # [batch, hid_size]

            elif self.encode_type == 'mean':
                token_embeddings = (
                    model_output.last_hidden_state
                )  # Contains all token embeddings
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                return torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def cos_sim(a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


def get_sentence_embeddings(
    model, input_ids, attention_mask, token_type_ids=None, encode_type='first_last_avg'
):
    """
    Returns the model output by encode_type as embeddings.
    """
    model_output = model(
        input_ids, attention_mask, token_type_ids, output_hidden_states=True
    )
    if encode_type == 'first_last_avg':
        first = model_output.hidden_states[1]
        last = model_output.hidden_states[-1]
        seq_length = first.size(1)  # Sequence length

        first_avg = torch.avg_pool1d(
            first.transpose(1, 2), kernel_size=seq_length
        ).squeeze(
            -1
        )  # [batch, hid_size]
        last_avg = torch.avg_pool1d(
            last.transpose(1, 2), kernel_size=seq_length
        ).squeeze(
            -1
        )  # [batch, hid_size]
        final_encoding = torch.avg_pool1d(
            torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(
                1, 2
            ),
            kernel_size=2,
        ).squeeze(-1)
        return final_encoding

    elif encode_type == 'last_avg':
        sequence_output = (
            model_output.last_hidden_state
        )  # [batch_size, max_len, hidden_size]
        seq_length = sequence_output.size(1)
        final_encoding = torch.avg_pool1d(
            sequence_output.transpose(1, 2), kernel_size=seq_length
        ).squeeze(-1)
        return final_encoding

    elif encode_type == 'cls':
        sequence_output = model_output.last_hidden_state
        return sequence_output[:, 0]  # [batch, hid_size]

    elif encode_type == 'pooler':
        return model_output.pooler_output  # [batch, hid_size]

    elif encode_type == 'mean':
        token_embeddings = (
            model_output.last_hidden_state
        )  # Contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
