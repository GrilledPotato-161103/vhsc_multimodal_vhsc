# tiny_vlm_jit.py
import math
import torch
import torch.nn as nn


class TinyVLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, n_classes=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        v = self.vision(image)                     # [B,d,H',W']
        b, d, hp, wp = v.shape
        v = v.flatten(2).transpose(1, 2)           # [B,S,d], S=H'*W'

        t = self.token_emb(input_ids)              # [B,T,d]

        q = t
        k = v
        val = v

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)  # [B,T,S]
        attn = torch.softmax(scores, dim=-1)                        # [B,T,S]
        ctx = torch.matmul(attn, val)                               # [B,T,d]

        pooled = ctx.mean(dim=1)                                    # [B,d]
        return self.head(pooled)                                    # [B,C]


def main():
    torch.manual_seed(0)
    model = TinyVLM(vocab_size=2000, d_model=64, n_classes=5).eval()

    B, H, W = 2, 128, 128
    T = 12
    image = torch.randn(B, 3, H, W, dtype=torch.float32)
    input_ids = torch.randint(0, model.vocab_size, (B, T), dtype=torch.int64)

    traced = torch.jit.trace(model, (image, input_ids), strict=True)
    traced = torch.jit.freeze(traced)

    print("\n=== TorchScript TRACED graph (inlined) ===")
    print(traced.inlined_graph)

    out = traced(image, input_ids)
    print("\nOutput shape:", tuple(out.shape))

    path = "data/tiny_vlm_traced.pt"
    traced.save(path)
    print("Saved TorchScript model:", path)


if __name__ == "__main__":
    main()
