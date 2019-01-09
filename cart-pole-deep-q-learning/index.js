/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * TensorFlow.js Reinforcement Learning Example: Balancing a Cart-Pole System.
 *
 * The simulation, training, testing and visualization parts are written
 * purely in JavaScript and can run in the web browser with WebGL acceleration.
 *
 * This reinforcement learning (RL) problem was proposed in:
 *
 * - Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
 *   Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
 *   Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983
 * - Sutton, "Temporal Aspects of Credit Assignment in Reinforcement Learning",
 *   Ph.D. Dissertation, Department of Computer and Information Science,
 *   University of Massachusetts, Amherst, 1984.
 *
 * It later became one of OpenAI's gym environmnets:
 *   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 */

import * as tf from '@tensorflow/tfjs';
import { sample } from 'underscore'

import {maybeRenderDuringTraining, onGameEnd, setUpUI} from './ui';

/**
 * Policy network for controlling the cart-pole system.
 *
 * The role of the policy network is to select an action based on the observed
 * state of the system. In this case, the action is the leftward or rightward
 * force and the observed system state is a four-dimensional vector, consisting
 * of cart position, cart velocity, pole angle and pole angular velocity.
 *
 */

// See https://keon.io/deep-q-learning/
class PolicyNetwork {
  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[] | tf.Model} hiddenLayerSizes
   *   Can be any of the following
   *   - Size of the hidden layer, as a single number (for a single hidden
   *     layer)
   *   - An Array of numbers (for any number of hidden layers).
   *   - An instance of tf.Model.
   */
  constructor(hiddenLayerSizesOrModel, input_shape, learningRate, decayRate) {
    this.memory = []
    this.gamma = 0.95;
    this.epsilon = 1.0;
    this.epsilon_min = 0.01;
    this.epsilon_decay = decayRate;
    this.learning_rate = learningRate;

    if (hiddenLayerSizesOrModel instanceof tf.Model) {
      this.model = hiddenLayerSizesOrModel;
    } else {
      this.model = this.createModel(hiddenLayerSizesOrModel, input_shape);
      this.target_model = this.createModel(hiddenLayerSizesOrModel, input_shape);    
    }
  }

  /**
   * Create the underlying model of this policy network.
   *
   * @param {number | number[]} hiddenLayerSizes Size of the hidden layer, as
   *   a single number (for a single hidden layer) or an Array of numbers (for
   *   any number of hidden layers).
   */
  createModel(hiddenLayerSizes, input_shape) {
    if (!Array.isArray(hiddenLayerSizes)) {
      hiddenLayerSizes = [hiddenLayerSizes];
    }
    const model = tf.sequential();
    hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
      model.add(tf.layers.dense({
        units: hiddenLayerSize,
        activation: 'relu',
        // `inputShape` is required only for the first layer.
        inputShape: i === 0 ? [4]: undefined
      }));
    });
    // The last layer has only one unit. The single output number will be
    // converted to a 0-1 for left and right
    model.add(tf.layers.dense({units: 2, activation:'linear'}));
    model.compile({
      loss: tf.losses.meanSquaredError,
      optimizer: tf.train.adam(this.learning_rate),
      metrics: [tf.metrics.meanAbsoluteError]
    });
    model.summary()
    return model
  }

  remember(state, action, reward, nextState, isDone) {
    if (this.memory.length > 1000000) {
      old = this.memory.shift()
      tf.dispose(old)
    }
    this.memory.push([state, action, reward, nextState, isDone])
  }

  async replay(batch_size) {
    const samples = sample(this.memory, batch_size);
    const losses = []
    for (let [state, action, reward, nextState, isDone] of samples) {
      const tensorQ = tf.tidy(() => {
        let target = tf.tensor(reward);
        if (!isDone) {
          const prediction = this.model.predict(nextState).dataSync()[0];
          target = tf.add(reward, tf.mul(tf.tensor(this.gamma), tf.max(prediction)));
        }
        const Q = this.model.predict(state).dataSync()
        Q[action] = target.dataSync();
        return tf.tensor(Q).expandDims(0);
      });
      losses.push((await this.model.trainOnBatch(state, tensorQ, { batchSize: 1 }))[0]);
    }
    if (this.epsilon > this.epsilon_min) {
      this.epsilon *= this.epsilon_decay
    }

    return tf.mean(losses).dataSync()
  }

  update_target_model() {
    this.target_model.setWeights(this.model.getWeights())
  }

  act(state, isTesting = false) {
    if (Math.random() <= this.epsilon && !isTesting) {
      return +(Math.random() < 0.5)
    } else {
      return tf.tidy(() => {
        const pred = this.model.predict(state).dataSync()
        return tf.argMax(pred)
      }).dataSync();
    }
  }

  /**
   * Train the policy network's model.
   *
   * @param {CartPole} cartPoleSystem The cart-pole system object to use during
   *   training.
   * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
   *   Optimizer to use for training.
   * @param {number} discountRate Reward discounting rate: a number between 0
   *   and 1.
   * @param {number} numGames Number of game to play for each model parameter
   *   update.
   * @param {number} maxStepsPerGame Maximum number of steps to perform during
   *   a game. If this number is reached, the game will end immediately.
   * @returns {number[]} The number of steps completed in the `numGames` games
   *   in this round of training.
   */
  async train(
      cartPoleSystem, numGames, maxStepsPerGame) {
    // TODO: Set batch size in UI
    const batch_size = 20
    const allRewards = [];
    const gameSteps = [];
    onGameEnd(0, numGames);
    for (let i = 0; i < numGames; ++i) {
      console.log(this.epsilon)
      // Randomly initialize the state of the cart-pole system at the beginning
      // of every game.
      cartPoleSystem.setRandomState();
      const gameRewards = [];
      const gameGradients = [];
      let gameLoss = []
      for (let j = 0; j < maxStepsPerGame; ++j) {
        const state = cartPoleSystem.getStateTensor();
        const action = this.act(state);
        const isDone = cartPoleSystem.update(action);
        const nextState = cartPoleSystem.getStateTensor();
        const reward = isDone ? -1 : 1;
        gameRewards.push(reward);
        this.remember(state, action, reward, nextState, isDone);
        if (isDone) break;
        if (this.memory.length > batch_size) {
          const history = await this.replay(batch_size);
          gameLoss.push(history)
        }
        await maybeRenderDuringTraining(cartPoleSystem);
      }
      if (gameLoss.length > 0) {
        console.log(`Loss: ${tf.mean(gameLoss).dataSync()}`)
      }

      gameSteps.push(gameRewards.length);
      onGameEnd(i+1, numGames);
      await tf.nextFrame();
    }
    this.update_target_model()
    return gameSteps;
  }

}

function reshape(arr, J, I) {
  const newArr = []
  for (let j = 0; j < J; ++j) {
    const tempArr = []
    for (let i = 0; i < I; ++i) {
      tempArr.push(arr[(i * I) + j])
    }
    newArr.push(tempArr)
  }
  return newArr
}

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1';

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
export class SaveablePolicyNetwork extends PolicyNetwork {
  /**
   * Constructor of SaveablePolicyNetwork
   *
   * @param {number | number[]} hiddenLayerSizesOrModel
   */
  constructor(hiddenLayerSizesOrModel, input_shape, learningRate, decayRate) {
    super(hiddenLayerSizesOrModel, input_shape, learningRate, decayRate);
  }

  /**
   * Save the model to IndexedDB.
   */
  async saveModel() {
    return await this.model.save(MODEL_SAVE_PATH_);
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns {SaveablePolicyNetwork} The instance of loaded
   *   `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
  static async loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`);
      const model = await tf.loadModel(MODEL_SAVE_PATH_);
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
      return new SaveablePolicyNetwork(model);
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }

  /**
   * Get the sizes of the hidden layers.
   *
   * @returns {number | number[]} If the model has only one hidden layer,
   *   return the size of the layer as a single number. If the model has
   *   multiple hidden layers, return the sizes as an Array of numbers.
   */
  hiddenLayerSizes() {
    const sizes = [];
    for (let i = 0; i < this.model.layers.length - 1; ++i) {
      sizes.push(this.model.layers[i].units);
    }
    return sizes.length === 1 ? sizes[0] : sizes;
  }
}

/**
 * Discount the reward values.
 *
 * @param {number[]} rewards The reward values to be discounted.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
 */
function discountRewards(rewards, discountRate) {
  const discountedBuffer = tf.buffer([rewards.length]);
  let prev = 0;
  for (let i = rewards.length - 1; i >= 0; --i) {
    const current = discountRate * prev + rewards[i];
    discountedBuffer.set(current, i);
    prev = current;
  }
  return discountedBuffer.toTensor();
}

/**
 * Discount and normalize reward values.
 *
 * This function performs two steps:
 *
 * 1. Discounts the reward values using `discountRate`.
 * 2. Normalize the reward values with the global reward mean and standard
 *    deviation.
 *
 * @param {number[][]} rewardSequences Sequences of reward values.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor[]} The discounted and normalize reward values as an
 *   Array of tf.Tensor.
 */
function discountAndNormalizeRewards(rewardSequences, discountRate) {
  return tf.tidy(() => {
    const discounted = [];
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }
    // Compute the overall mean and stddev.
    const concatenated = tf.concat(discounted);
    const mean = tf.mean(concatenated);
    const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
    // Normalize the reward sequences using the mean and std.
    const normalized = discounted.map(rs => rs.sub(mean).div(std));
    return normalized;
  });
}

/**
 * Scale the gradient values using normalized reward values and compute average.
 *
 * The gradient values are scaled by the normalized reward values. Then they
 * are averaged across all games and all steps.
 *
 * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
 *   name to all the gradient values for the variable across all games and all
 *   steps.
 * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
 *   for all the games. Each element of the Array is a 1D tf.Tensor of which
 *   the length equals the number of steps in the game.
 * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
 *   for the variables.
 */
function scaleAndAverageGradients(allGradients, normalizedRewards) {
  return tf.tidy(() => {
    const gradients = {};
    for (const varName in allGradients) {
      gradients[varName] = tf.tidy(() => {
        // Stack gradients together.
        const varGradients = allGradients[varName].map(
            varGameGradients => tf.stack(varGameGradients));
        // Expand dimensions of reward tensors to prepare for multiplication
        // with broadcasting.
        const expandedDims = [];
        for (let i = 0; i < varGradients[0].rank - 1; ++i) {
          expandedDims.push(1);
        }
        const reshapedNormalizedRewards = normalizedRewards.map(
            rs => rs.reshape(rs.shape.concat(expandedDims)));
        for (let g = 0; g < varGradients.length; ++g) {
          // This mul() call uses broadcasting.
          varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
        }
        // Concatenate the scaled gradients together, then average them across
        // all the steps of all the games.
        return tf.mean(tf.concat(varGradients, 0), 0);
      });
    }
    return gradients;
  });
}

setUpUI();
