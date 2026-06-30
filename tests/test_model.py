import torch


def test_model_forward_cpu():
    from evidential_dualunet.models import build_model
    cfg = {"model": {"model_name": "dualunet", "encoder_name": "resnet18", "encoder_weights": None, "classes_s": 2, "classes_c": 1, "decoder_channels": [64, 32, 16, 8, 4], "decoder_use_batchnorm": True}}
    model = build_model(cfg).eval()
    with torch.inference_mode(): output = model(torch.randn(1, 3, 64, 64))
    assert output["seg"]["alpha"].shape == (1, 3, 64, 64)
    assert torch.allclose(output["seg"]["p_hat"].sum(1), torch.ones(1, 64, 64), atol=1e-5)

