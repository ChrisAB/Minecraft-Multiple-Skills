from neural_networks.q_network import QNetwork
import h5py
import argparse
import torch


if __name__ == "__main__":

    qNetwork = QNetwork(100, 64, 5, -1)
    rate = torch.optim.lr_scheduler.StepLR(optimizer=torch.optim.Adam(
        params=qNetwork.parameters(), lr=0.0005), step_size=250, gamma=0.9999)
    print(qNetwork)

    hiddenWidth1 = 100
    hiddenWidth2 = 64
    outputWidth = 5
    weightInit = -1
    batchSize = 4
    gamma = 0.7

    dataOut = h5py.File('skillWeightsQ.h5', 'w')

    print('Loading data...')
    # data = h5py.File(FLAGS.file, 'r')
    # numSkills = data.get('numberSkills')
    numSkills = 4
    print('Number of skills is ' + str(numSkills))

    dataOut.create_dataset('hiddenWidth', data=hiddenWidth1)
    dataOut.create_dataset('numberSkills', data=numSkills)

    for skill in range(numSkills):
        activations = np.array(data.get('activations_' + str(skill)))
        actions = (np.array(data.get('actions_' + str(skill))) - 1)
        termination = np.array(data.get('termination_' + str(skill)))

        print('Creating model...')
        qNetwork = QNetwork(100, 64, 5, -1)
        optimizer = torch.optim.lr_scheduler.StepLR(optimizer=torch.optim.Adam(
            params=qNetwork.parameters(), lr=0.0005), step_size=250, gamma=0.9999)
        maxQ = 1
        iterations = 0
        for _ in range(20000000):
            if (_ % 1000000 == 0 and _ > 0):
                testPredictions = qNetwork.predict(activations[int(
                    math.ceil(activations.shape[0] * 0.8)) + 1:activations.shape[0], :])
                trainPredictions = qNetwork.predict(
                    activations[0:int(math.ceil(activations.shape[0] * 0.8)), :])
                print('Done ' + _ + ' iterations. testing error is:...')
            print('Loss: ' + loss_val + ', Skill#: ' + skill)
        index = np.random.randint(
            int(math.ceil(activations.shape[0] * 0.8)), size=batchSize)

        allQ = qNetwork.predict(activations[index, :])
        Q1 = qNetwork.predict(activations[index+1, :])

        targetQ = np.ones(allQ.shape) * -1

        for i in range(index.shape[0]):
            if termination[index[i]] == 1:
                Q = 0
            else:
                Q = np.max(Q1[i, :]) * gamma

            # maxQ = max(maxQ, abs(Q))
            targetQ[i, :] = targetQ[i, :] + Q - gamma * gamma
            targetQ[i, int(actions[index[i]])] = targetQ[i,
                                                         int(actions[index[i]])] + gamma * gamma
        targetQ = targetQ * 1.0 / maxQ

        loss = qNetwork.loss(activations[index, :], targetQ)

        loss.backward()

        optimizer.step()

        _, loss_val = sess.run([updateModel, loss], feed_dict={
                               x: activations[index, :], nextQ: targetQ})
