from q_network import QNetwork
import h5py
import argparse


if __name__ == "__main__":

    qNetwork = QNetwork(100, 64, 5, -1)
    rate = torch.optim.lr_scheduler.StepLR(optimizer=torch.optim.Adam(
        params=qNetwork.parameters(), lr=0.0005), step_size=250, gamma=0.9999)
    print(qNetwork)

    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, required=True,
                        help='Name of Skill extraction file.')
    FLAGS, unparsed = parser.parse_known_args()

    hiddenWidth1 = 100
    hiddenWidth2 = 64
    outputWidth = 5
    weightInit = -1
    batchSize = 4
    gamma = 0.7

    dataOut = h5py.File('skillWeightsQ.h5', 'w')

    print('Loading data...')
    #data = h5py.File(FLAGS.file, 'r')
    #numSkills = data.get('numberSkills')
    print('Number of skills is ' + str(numSkills[()]))

    dataOut.create_dataset('hiddenWidth', data=hiddenWidth1)
    dataOut.create_dataset('numberSkills', data=numSkills)
    '''
    for skill in range(numSkills[()]):
        activations = np.array(data.get('activations_' + str(skill)))
        actions = (np.array(data.get('actions_' + str(skill))) - 1)
        termination = np.array(data.get('termination_' + str(skill)))

        print('Creating model...')
        qNetwork = QNetwork(100, 64, 5, -1)
        rate = torch.optim.lr_scheduler.StepLR(optimizer=torch.optim.Adam(
            params=qNetwork.parameters(), lr=0.0005), step_size=250, gamma=0.9999)
            '''
