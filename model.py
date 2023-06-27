import segmentation_models_pytorch as smp

def unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None):
    # Create a U-Net model using the specified encoder
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, activation=activation)

    # Freeze the parameters of the encoder up to a specific layer
    # In this case, the layer 'unet.encoder.layer4.2.bn2.bias' is the breaking point
    # Any parameters before this layer will be frozen and not updated during training
    for name, param in model.named_parameters():
        if name == 'unet.encoder.layer4.2.bn2.bias':
            break
        param.requires_grad = False

    return model