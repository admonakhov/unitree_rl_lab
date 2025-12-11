import pickle
import numpy as np
import argparse
# import torch 

parser = argparse.ArgumentParser(description="Replay motion from pkl file and output to csv file.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="The path to the input motion pkl file.")
parser.add_argument("--output_name", type=str, help="The name of the motion csv file.")
args_cli = parser.parse_args()


if not args_cli.output_name:
    # generate at the same location as input file
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1]) + "/" + args_cli.input_file.split("/")[-1].replace(".pkl", ".csv")
    )

print(args_cli.input_file)

with open(args_cli.input_file, 'rb') as f:
    df = pickle.load(f)
# vel = torch.gradient(torch.tensor(df['root_pos']), spacing=1/df['fps'], dim=0)[0].numpy()
out = np.concat([df['root_pos'], df['root_rot'],   df['dof_pos']], axis=1)
np.savetxt(args_cli.output_name, out, delimiter=',')