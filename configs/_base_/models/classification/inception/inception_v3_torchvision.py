model = dict(
    type='inception_v3_torchvision',
    progress=True,
    # weights=None,
    num_classes=1000,
    aux_logits=False,
    transform_input=False,
    dropout=0.5
)
