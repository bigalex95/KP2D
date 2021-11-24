import torch
import argparse
from kp2d.networks.keypoint_net_mmvc import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet

def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--device", default='cuda', type=str, help="Choose computing device (cpu/cuda)")

    args = parser.parse_args()
    cmpDevice = torch.device(args.device)
    checkpoint = torch.load(args.pretrained_model, map_location=args.device)
    model_args = checkpoint['config']['model']['params']

    # Check model type
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        net_type = checkpoint['config']['model']['params']
    else:
        net_type = KeypointNet # default when no type is specified

    # Create and load keypoint net
    if net_type is KeypointNet:
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'])
    else:
        keypoint_net = KeypointResnet()

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    
    if torch.cuda.is_available():
        keypoint_net = keypoint_net.to(cmpDevice)

    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    # ONNX
    dummy_input = torch.randn(1, 3, 240, 320)
    input_names = [ "input", "grid" ]
    output_names = [ "output" ]

    print("ONNX converter")

    torch.onnx.export(keypoint_net, 
                  dummy_input,
                  "./data/models/onnx/keypointNetV4.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  opset_version=11
                  )
    
if __name__ == '__main__':
    main()