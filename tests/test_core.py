import torch

from handwriting_var.auxiliary import WordOCR, freeze_auxiliary
from handwriting_var.conditioning import StyleEncoder
from handwriting_var.config import ModelConfig
from handwriting_var.tokenizer import MultiScaleVQVAE
from handwriting_var.var import ConditionalVAR, block_causal_mask, categorical_straight_through, scale_slices


def test_token_shapes_and_count():
    model=MultiScaleVQVAE(latent_dim=32,codebook_size=64)
    reconstruction,tokens,*_=model(torch.randn(1,1,64,256))
    assert reconstruction.shape==(1,1,64,256)
    assert [tuple(x.shape[1:]) for x in tokens]==list(ModelConfig().scales)
    assert sum(x[0].numel() for x in tokens)==123


def test_scale_causal_mask_has_no_future_leak():
    mask=block_causal_mask(); slices=scale_slices()
    for current,row in enumerate(slices):
        for future,col in enumerate(slices):
            assert bool(mask[row,col].all()) is (future>current)


def test_style_requires_exactly_one_reference():
    model=StyleEncoder(dimension=48,layers=1,heads=6)
    assert model(torch.randn(2,1,64,256)).shape==(2,4,48)
    try: model(torch.randn(2,3,1,64,256))
    except ValueError: pass
    else: raise AssertionError("multiple references must be rejected")


def test_st_bridge_and_frozen_auxiliary_propagate_to_logits():
    tokenizer=MultiScaleVQVAE(latent_dim=32,codebook_size=32).freeze()
    logits=[torch.randn(2,h,w,32,requires_grad=True) for h,w in tokenizer.quantizer.scales]
    image=categorical_straight_through(logits,tokenizer)
    auxiliary=freeze_auxiliary(WordOCR(classes=10,width=64,depth=1,heads=8))
    auxiliary(image).mean().backward()
    assert all(x.grad is not None and x.grad.abs().sum()>0 for x in logits)
    assert all(x.grad is None for x in tokenizer.parameters())
    assert all(x.grad is None for x in auxiliary.parameters())


def test_zero_initialized_var_gates():
    model=ConditionalVAR(codebook_size=32,latent_dim=32,dimension=48,heads=6,layers=2,gradient_checkpointing=False)
    assert all(torch.count_nonzero(block.gates)==0 for block in model.blocks)

