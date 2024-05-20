import torch
import MDAnalysis as mda
import numpy as np

# Load the universe
u = mda.Universe('drug_151_no_solvent.nc')
u1 = mda.Universe('drug_151_h.pdb')
# Set the number of frames to save
num_frames = 5000

# Create lists to store positions, velocities, and atom types
pos = []
vel = []
atom_types = []

# Iterate over the trajectory
for i, ts in enumerate(u.trajectory):
    # Append positions, velocities, and atom types to the lists
    pos.append(torch.tensor(ts._pos))
    atom_types.append(u1.atoms.types)

    if i < num_frames - 1:
        vel.append(torch.tensor(u.trajectory[i + 1]._pos) - torch.tensor(ts._pos))

    # Break the loop if the desired number of frames is reached
    if i == num_frames - 1:
        break

# Convert the lists to tensors
pos = torch.stack(pos)
vel = torch.stack(vel)

# Get the unique atom types and assign class IDs
# atom_class = {}
# atoms = []
# for atom_type in u1.atoms.types:
#     if atom_type not in atom_class:
#         atoms.append(len(atom_class) + 1)
#         atom_class[atom_type] = len(atom_class) + 1
#     else:
#         atoms.append(atom_class[atom_type])


atom_class = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br':  8, 'I': 9, 'Cl':10, '<unk>':11}
atoms = []
for atom_type in u1.atoms.types:
    if atom_type not in atom_class:
        atoms.append(atom_class['<unk>'])
    else:
        atoms.append(atom_class[atom_type])

atoms = torch.tensor(atoms)

# Save the data
data = {
    'atoms': atoms.numpy(),
    'pos': pos.numpy(),
    'vel': vel.numpy(),
    'atom_types': np.array(atom_types)
}
print(data)

np.save('md/drug_151_5K.npy', data)
