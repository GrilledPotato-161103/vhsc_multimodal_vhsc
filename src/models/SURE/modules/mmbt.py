import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel

from models.SURE.modules.mmbt_image import ImageEncoder


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class UnimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(UnimodalBertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def forward(self, input_txt=None, attention_mask=None, segment=None, input_img=None):
        # import IPython; IPython.embed(); exit(1)
        bsz = input_txt.size(0) if input_txt is not None else input_img.size(0)
        attention_mask = torch.cat(
                [
                    torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                    attention_mask,
                ],
                dim=1,
            )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        txt_embed_out = None
        img_embed_out = None
        if input_txt is not None:
            txt_embed_out = self.txt_embeddings(input_txt, segment)
        
        if input_img is not None:
            img_tok = (
                torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
                .fill_(0)
                .cuda()
            )
            img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
            img_embed_out = self.img_embeddings(img, img_tok)
        
        if txt_embed_out is None:
            txt_embed_out = torch.zeros_like(img_embed_out)
        if img_embed_out is None:
            img_embed_out = torch.zeros_like(txt_embed_out)
        
        return img_embed_out, txt_embed_out, extended_attention_mask

class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        if args.task == "vsnli":
            ternary_embeds = nn.Embedding(3, args.hidden_sz)
            ternary_embeds.weight.data[:2].copy_(
                bert.embeddings.token_type_embeddings.weight
            )
            ternary_embeds.weight.data[2].copy_(
                bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)
            )
            self.txt_embeddings.token_type_embeddings = ternary_embeds

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        encoded_layers = self.encoder(
            encoder_input, extended_attention_mask, output_all_encoded_layers=False
        )

        return self.pooler(encoded_layers[-1])

class MultimodalBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment, img):
        x = self.enc(txt, mask, segment, img)
        return self.clf(x)

class Reconstructor(nn.Module):
    def __init__(self, in_channels, out_channels, common_dim, latent_dim, positive=False):
        super(Reconstructor, self).__init__()
        self.positive = positive

        # if positive:
        #     self.encode = nn.Sequential(
        #                     nn.Conv1d(in_channels, 1, 1),
        #                     nn.ReLU(),
        #                     nn.Linear(common_dim, 1),
        #                     )
        # else:
        self.encode = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, 1),
                        nn.ReLU(),
                        nn.Linear(common_dim, latent_dim),
                        )

    def forward(self, x):
        x = self.encode(x)
        if self.positive:
            x = torch.log(1 + torch.exp(x))
        return x

class ReconstructUncertainty(nn.Module):
    def __init__(self, in_channels, out_channels, common_dim, latent_dim):
        super(ReconstructUncertainty, self).__init__()

        self.encode = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, 1),
                        nn.ReLU(),
                        nn.Linear(common_dim, latent_dim),
                        )
        
        self.fuse_uncert = nn.Sequential(
                        nn.Conv1d(out_channels * 2, 1, 1),
                        nn.ReLU(),
                        nn.Linear(latent_dim, 1),
                        )

    def forward(self, x, muy):
        x = self.encode(x)
        x = torch.cat([x, muy], dim=1)
        x = self.fuse_uncert(x)
        x = F.softplus(x)
        return x

class OutputSigma2(nn.Module):
    def __init__(self, in_dim, n_classes, out_dim):
        super(OutputSigma2, self).__init__()
        self.net_1 = nn.Sequential(
            nn.Linear(in_dim + n_classes, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim)
        )

    def forward(self, x, rec):
        x_ = self.net_1(torch.cat((x, rec), dim=-1)) + x
        out = self.net_2(x_)
        out = torch.log(1 + torch.exp(out))
        return out

if __name__ == '__main__':
    a = torch.randn(32, 5, 768)
    rec = Reconstructor(5, 21, 768, 128)
    print(rec(a).size())