let input = [];
let weight1 = [];
let weight2 = [];
let new_weight1 = [];
let new_weight2 = [];
let hidden = [];
let weight3 = [];
let new_weight3 = [];
let new_weight4 = [];
let weight4 = [];
let output = [];
let b1;
let b2;

let net_input_for_hidden1;
let net_input_for_hidden2;

let net_output_of_hidden1;
let net_output_of_hidden2;

let net_input_for_output1;
let net_input_for_output2;

let net_output_of_output1;
let net_output_of_output2;

let target_output1= 0.01;
let target_output2= 0.99;

let total_error;
let delta_total_error_wrt_weight1_1;
let delta_total_error_wrt_weight1_2;
let delta_total_error_wrt_weight2_1;
let delta_total_error_wrt_weight2_2;
let delta_total_error_wrt_weight3_1;
let delta_total_error_wrt_weight3_2;
let delta_total_error_wrt_weight4_1;
let delta_total_error_wrt_weight4_2;

let learning_rate = 0.5;

input[1] = 0.05;
input[2] = 0.1;

weight1[1] = 0.15;
weight1[2] = 0.25;

weight2[1] = 0.2;
weight2[2] = 0.3;

weight3[1] = 0.4;
weight3[2] = 0.5;

weight4[1] = 0.45;
weight4[2] = 0.55;

b1 = 0.35;
b2 = 0.6;

for(_i = 1; _i < 5000; _i++){
    net_input_for_hidden1 = weight1[1] * input[1] + weight2[1] * input[2] + b1 * 1;
    console.log({net_input_for_hidden1});
    net_output_of_hidden1 = 1 / (1 + Math.exp(-net_input_for_hidden1));
    console.log({net_output_of_hidden1});

    net_input_for_hidden2 = weight1[2] * input[1] + weight2[2] * input[2] + b1 * 1;
    console.log({net_input_for_hidden2});
    net_output_of_hidden2 = 1 / (1 + Math.exp(-net_input_for_hidden2));
    console.log({net_output_of_hidden2});

    net_input_for_output1 = net_output_of_hidden1 * weight3[1] + net_output_of_hidden2 * weight4[1] + b2 * 1;
    console.log({net_input_for_output1});
    net_output_of_output1 = 1 / (1 + Math.exp(-net_input_for_output1));
    console.log({net_output_of_output1});

    net_input_for_output2 = net_output_of_hidden1 * weight3[2] + net_output_of_hidden2 * weight4[2] + b2 * 1;
    console.log({net_input_for_output2});
    net_output_of_output2 = 1 / (1 + Math.exp(-net_input_for_output2));
    console.log({net_output_of_output2});

    total_error =  (Math.pow((target_output1 - net_output_of_output1), 2) / 2 )+  (Math.pow((target_output2 - net_output_of_output2),2) /2 );

    console.log({total_error});

    /* OUTPUT LAYER BACKPROPAGATION */

    /*

    EQN 1: Partial derivative of total_error wrt net_output_of_output1 equals partial derivative of SUMMATION[1/2 * (target - output)^2]
    We have 2 output neruons, so we have 2 targets and 2 outputs

    SUMMATION[1/2 * (target - output)^2] = [1/2 * (target_output1 - net_output_of_output1)^2] + [1/2 * (target_output2 - net_output_of_output2)^2]
    Partial derivative of the above wrt to net_output_of_output1 (here net_output_of_output2 is constant) = [2 * 1/2 * (target_output1 - net_output_of_output1) * -1]
    So, Partial derivate of total_error wrt net_output_of_output1 = net_output_of_output1 - target_output1
    total_error / net_output_of_output1 = net_output_of_output1 - target_output1

    Similarly, partial derivate of total_error wrt net_output_of_output2 = net_output_of_output2 - target_output2

    */

    /*

    EQN 2: Partial derivative of net_output of a neuron wrt net_input of the same neuron is a Logistic Function [1/(1 + e^x)]
    For output neuron 1, we have the following:

    net_output_of_output1 / net_input_of_output1 = Partial derivate of net_output_of_output1 wrt net_input_of_output1
    Using Logistic Function to squash the equation,
    net_output_of_output1 / net_input_of_output1 = Partial derivate of ( 1 / 1 + e^net_input_of_output1)
    net_output_of_output1 / net_input_of_output1 = net_output_of_output1 * (1 - net_output_of_output1)

    Similarly, Partial derivate of net_output_of_output2 wrt net_input_of_output2 = net_output_of_output2 * (1 - net_output_of_output2)

    */

    /*

    EQN 3: Partial derivate of net_input_for_output1 wrt weight3[1]
    net_input_for_output1 = net_output_of_hidden1 * weight3[1] + net_output_of_hidden2 * weight4[1]
    net_input_for_output1 wrt weight3[1] = net_output_of_hidden1 + 0 = net_output_of_hidden1

    Its same for partial derivatives of all inputs wrt to respective weights

    */

    /*
        total_error / weight3[1] = (total_error / net_output_of_output1) * (net_output_of_output1 / net_input_for_output1) * (net_input_for_output1 / weight3[1])
        
        total_error / net_output_of_output1 = net_output_of_output1 - target_outpu1; (EQN 1)
        net_output_of_output1 / net_input_for_output1 = net_output_of_output1 * (1 - net_output_of_output1); (EQN 2)
        net_input_for_output1 / weight3[1] = net_output_of_hidden1; (EQN 3)

        delta_total_error_wrt_weight3_1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * net_output_of_hidden1;
    */

    delta_total_error_wrt_weight3_1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * net_output_of_hidden1;
    console.log({delta_total_error_wrt_weight3_1});
    new_weight3[1] = weight3[1] - (learning_rate * delta_total_error_wrt_weight3_1);
    console.log({new_weight3});

    /*
        total_error / weight4[1] = (total_error / net_output_of_output1) * (net_output_of_output1 / net_input_for_output1) * (net_input_for_output1 / weight4[1])
        
        total_error / net_output_of_output1 = net_output_of_output1 - target_outpu1;
        net_output_of_output1 / net_input_for_output1 = net_output_of_output1 * (1 - net_output_of_output1);
        net_input_for_output1 / weight4[1] = net_output_of_hidden2;

        delta_total_error_wrt_weight4_1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * net_output_of_hidden2;
    */

    delta_total_error_wrt_weight4_1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * net_output_of_hidden2;
    console.log({delta_total_error_wrt_weight4_1});
    new_weight4[1] = weight4[1] - (learning_rate * delta_total_error_wrt_weight4_1);
    console.log({new_weight4});

    /*
        total_error / weight3[2] = (total_error / net_output_of_output2) * (net_output_of_output2 / net_input_for_output2) * (net_input_for_output2 / weight3[2])
        
        total_error / net_output_of_output2 = net_output_of_output2 - target_outpu2;
        net_output_of_output2 / net_input_for_output2 = net_output_of_output2 * (1 - net_output_of_output2);
        net_input_for_output2 / weight3[2] = net_output_of_hidden1;

        delta_total_error_wrt_weight3_2 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * net_output_of_hidden1;
    */

    delta_total_error_wrt_weight3_2 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * net_output_of_hidden1;
    console.log({delta_total_error_wrt_weight3_2});
    new_weight3[2] = weight3[2] - (learning_rate * delta_total_error_wrt_weight3_2);
    console.log({new_weight3});

    /*
        total_error / weight4[2] = (total_error / net_output_of_output2) * (net_output_of_output2 / net_input_for_output2) * (net_input_for_output2 / weight4[2])
        
        total_error / net_output_of_output2 = net_output_of_output2 - target_outpu2;
        net_output_of_output2 / net_input_for_output2 = net_output_of_output2 * (1 - net_output_of_output2);
        net_input_for_output2 / weight4[2] = net_output_of_hidden2;

        delta_total_error_wrt_weight4_2 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * net_output_of_hidden2;
    */

    delta_total_error_wrt_weight4_2 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * net_output_of_hidden2;
    console.log({delta_total_error_wrt_weight4_2});
    new_weight4[2] = weight4[2] - (learning_rate * delta_total_error_wrt_weight4_2);
    console.log({new_weight4});

    /* HIDDEN LAYER BACKPROPAGATION */

    /*
        CALCULATE NEW WEIGHT1[1]
        
        (a) total_error / weight1[1] = (b1) (total_error / net_output_of_hidden1) * (b2) (net_output_of_hidden1 / net_input_for_hidden1) * (b3) (net_input_for_hidden1 / weight1[1])
        
        (b1) total_error / net_output_of_hidden1 = (total_error_output1 + total_error_output2) / net_output_of_hidden1
        (b1) total_error / net_output_of_hidden1 = (c) total_error_output1 / net_output_of_hidden1 + (d) total_error_output2 / net_output_of_hidden1
        
        (c) total_error_output1 / net_output_of_hidden1 = (e) total_error_output1 / net_input_for_output1 * (f) net_input_for_output1 / net_output_of_hidden1;
        
        NOTE: Partial derivative of total_error_output1 wrt to net_input_for_output1 is equal to partial derivative of total_error_output1 wrt net_output_of_output1 * partial derivative of net_output_of_output1 wrt net_input_for_output1;
        (e) total_error_output1 / net_input_for_output1 = total_error_output1 / net_output_of_output1 * net_output_of_output1 / net_input_for_output1;
        Using EQN 1, partial derivative of total_error_output1 wrt net_output_of_output1 is equal to net_output_of_output1 - target_output1 (because all others are constant)
        Using EQN 2, partial derivative of net_input_for_output1 wrt net_output_of_hidden1 equals net_output_of_hidden1 * (1 - net_output_of_hidden1)
        
        So, total_error_output1 / net_output_of_output1  = (net_output_of_output1 - target_output1) (EQN 1)
        net_output_of_output1 / net_input_for_output1 = net_output_of_output1 * (1 - net_output_of_output1) (EQN 2)

        total_error_output1 / net_input_for_output1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1)
        Now (e) is solved

        Next, we need to calculate net_input_for_output1 / net_output_of_hidden1 (partial derivative of net_input_for_output1 wrt net_output_of_hidden1)
        net_input_for_output1 = net_output_of_hidden1 * weight3[1] + net_output_of_hidden2 * weight4[1]
        Partial derivative of net_input_for_output1 wrt net_output_of_hidden1 = weight3[1] (all others will be constants)
        So, net_input_for_output1 / net_output_of_hidden1 = weight3[1]
        Now, (f) is solved

        Finally, (c) can be calculated as (e) * (f):
        (c) total_error_output1 / net_output_of_hidden1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * weight3[1]
        So, (g) var delta_total_error_output1_wrt_net_output_of_hidden1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * weight3[1]

        a = b1 * b2 * b3
        a = (c + d) * b2 * b3
        a = e * f * b2 * b3 + d * b2 * b3
        a = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * weight3[1] * b2 * b3 + d * b2 * b3;
    */

    delta_total_error_output1_wrt_net_output_of_hidden1 = (net_output_of_output1 - target_output1) * net_output_of_output1 * (1 - net_output_of_output1) * weight3[1]
    console.log("g", delta_total_error_output1_wrt_net_output_of_hidden1);

    /*
        Now we calculate (d), in the same way as (c)
        (d) total_error_output2 / net_output_of_hidden1 = (h) total_error_output2 / net_output_of_output2 * (i) net_output_of_output2 / net_input_for_output2 * (j) net_input_for_output2 / net_output_of_hidden1;
        
        (h) total_error_output2 / net_output_of_output2 = net_output_of_output2 - target_output2 (EQN 1)
        (i) net_output_of_output2 / net_input_for_output2 = net_output_of_output2 * (1 - net_output_of_output2) (EQN 2)
        (j) net_input_for_output2 / net_output_of_hidden1 is calculated as follows:
        net_input_for_output2 = net_output_of_hidden1 * weight3[2] + net_output_of_hidden2 * weight4[2] + b2 * 1
        Partial derivative of net_input_for_output2 wrt net_output_of_hidden1 = weight3[2] (because all others will be constants)
        So, (j) net_input_for_output2 / net_output_of_hidden1 = weight3[2]

        Now, (d) is calculated as:
        (d) total_error_output2 / net_output_of_hidden1 = (h) total_error_output2 / net_output_of_output2 * (i) net_output_of_output2 / net_input_for_output2 * (j) net_input_for_output2 / net_output_of_hidden1;
        (d) total_error_output2 / net_output_of_hidden1 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * weight3[2]

        So, (k) var delta_total_error_output2_wrt_net_output_of_hidden1 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * weight3[2];
    */

    var delta_total_error_output2_wrt_net_output_of_hidden1 = (net_output_of_output2 - target_output2) * net_output_of_output2 * (1 - net_output_of_output2) * weight3[2];
    console.log("k", delta_total_error_output2_wrt_net_output_of_hidden1);

    /*
        We now have (c) and (d)
        (b1) = (c) + (d)
        (l) var delta_total_error_wrt_net_output_of_hidden1 = delta_total_error_output1_wrt_net_output_of_hidden1 + delta_total_error_output2_wrt_net_output_of_hidden1
    */

    var delta_total_error_wrt_net_output_of_hidden1 = delta_total_error_output1_wrt_net_output_of_hidden1 + delta_total_error_output2_wrt_net_output_of_hidden1
    console.log("l", delta_total_error_wrt_net_output_of_hidden1);

    /*
        Next we calculate b2 and b3
        (b2) = net_output_of_hidden1 / net_input_for_hidden1
        Using EQN 2, net_output_of_hidden1 / net_input_for_hidden1 = net_output_of_hidden1 * (1 - net_output_of_hidden1)
        So, (m) var delta_net_output_of_hidden1_wrt_net_input_for_hidden1 = net_output_of_hidden1 * (1 - net_output_of_hidden1)

        (b3) = net_input_for_hidden1 / weight1[1]
        net_input_for_hidden1 = weight1[1] * input[1] + weight2[1] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden1 wrt weight1[1] = input[1]
        So, (n) var delta_net_input_for_hidden1_wrt_weight1_1 = input[1]
    */

    var delta_net_output_of_hidden1_wrt_net_input_for_hidden1 = net_output_of_hidden1 * (1 - net_output_of_hidden1);
    var delta_net_input_for_hidden1_wrt_weight1_1 = input[1];

    console.log("m", delta_net_output_of_hidden1_wrt_net_input_for_hidden1);
    console.log("n", delta_net_input_for_hidden1_wrt_weight1_1);

    /*
        Finally, we calculate (a) = b1 * b2 * b3
        (o) var delta_total_error_wrt_weight1_1 = delta_total_error_wrt_net_output_of_hidden1 * delta_net_output_of_hidden1_wrt_net_input_for_hidden1 * delta_net_input_for_hidden1_wrt_weight1_1;
    */

    delta_total_error_wrt_weight1_1 = delta_total_error_wrt_net_output_of_hidden1 * delta_net_output_of_hidden1_wrt_net_input_for_hidden1 * delta_net_input_for_hidden1_wrt_weight1_1;

    console.log("o", delta_total_error_wrt_weight1_1);

    new_weight1[1] = weight1[1] - (learning_rate * delta_total_error_wrt_weight1_1);
    console.log({new_weight1});

    /*
        CALCULATE NEW WEIGHT1[2]

        (2a1) total_error / weight1[2] = (2b) (total_error_output1 / net_output_of_output1) * (2c) (net_output_of_output1 / net_input_for_output1) * (2d) (net_input_for_output1 / net_output_of_hidden2) * (2e) (net_output_of_hidden2 / net_input_for_hidden2) * (2f) (net_input_for_hidden2 / weight1[2])

        (2b) total_error_output1 / net_output_of_output1 = net_output_of_output1 - target_output1; (EQN 1)
        (2c) net_output_of_output1 / net_input_for_output1 = net_output_of_output1 * (1 - net_output_of_output1) (EQN 2)
        
        (2d) net_input_for_output1 / net_output_of_hidden2 is calculated as follows:
        net_input_for_output1 = net_output_of_hidden1 * weight3[1] + net_output_of_hidden2 * weight4[1] + b2 * 1;
        Partial derivative of net_input_for_output1 wrt net_output_of_hidden2 = weight4[1]

        (2e) net_output_of_hidden2 / net_input_for_hidden2 = net_output_of_hidden2 * (1 - net_output_of_hidden2);

        (2f) net_input_for_hidden2 / weight1[2] is calculated as follows:
        net_input_for_hidden2 = weight1[2] * input[1] + weight2[2] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden2 wrt weight1[2] = input[1]

        2a1 = 2b * 2c * 2d * 2e * 2f
        var _2a1 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight4[1]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[1]);

        (2a2) total_error / weight1[2] = (3b) (total_error_output2 / net_output_of_output2) * (3c) (net_output_of_output2 / net_input_for_output2) * (3d) (net_input_for_output2 / net_output_of_hidden2) * (3e) (net_output_of_hidden2 / net_input_for_hidden2) * (3f) (net_input_for_hidden2 / weight1[2])

        (3b) total_error_output2 / net_output_of_output2 = net_output_of_output2 - target_output2; (EQN 1)
        (3c) net_output_of_output2 / net_input_for_output2 = net_output_of_output2 * (1 - net_output_of_output2) (EQN 2)
        
        (3d) net_input_for_output2 / net_output_of_hidden2 is calculated as follows:
        net_input_for_output2 = net_output_of_hidden1 * weight3[2] + net_output_of_hidden2 * weight4[2] + b2 * 1;
        Partial derivative of net_input_for_output2 wrt net_output_of_hidden2 = weight4[2]

        (3e) net_output_of_hidden2 / net_input_for_hidden2 = net_output_of_hidden2 * (1 - net_output_of_hidden2);

        (3f) net_input_for_hidden2 / weight1[2] is calculated as follows:
        net_input_for_hidden2 = weight1[2] * input[1] + weight2[2] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden2 wrt weight1[2] = input[1]

        2a2 = 3b * 3c * 3d * 3e * 3f
        var _2a2 = (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight4[2]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[1]);

        (p) delta_total_error_wrt_weight1_2 = _2a1 + _2a2;
    */

    var _2a1 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight4[1]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[1]);
    var _2a2 = (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight4[2]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[1]);
    delta_total_error_wrt_weight1_2 = _2a1 + _2a2;
    console.log("p", delta_total_error_wrt_weight1_2);

    new_weight1[2] = weight1[2] - (learning_rate * delta_total_error_wrt_weight1_2);
    console.log({new_weight1});

    /*
        CALCULATE NEW WEIGHT2[1]

        total_error / weight2[1] = (2a1) + (2a2)

        (2a1) = (2b) (total_error_output1 / net_output_of_output1) * (2c) (net_output_of_output1 / net_input_for_output1) * (2d) (net_input_for_output1 / net_output_of_hidden1) * (2e) (net_output_of_hidden1 / net_input_for_hidden1) * (2f) (net_input_for_hidden1 / weight2[1])

        (2b) total_error_output1 / net_output_of_output1 = net_output_of_output1 - target_output1; (EQN 1)
        (2c) net_output_of_output1 / net_input_for_output1 = net_output_of_output1 * (1 - net_output_of_output1) (EQN 2)
        
        (2d) net_input_for_output1 / net_output_of_hidden1 is calculated as follows:
        net_input_for_output1 = net_output_of_hidden1 * weight3[1] + net_output_of_hidden2 * weight4[1] + b2 * 1;
        Partial derivative of net_input_for_output1 wrt net_output_of_hidden1 = weight3[1]

        (2e) net_output_of_hidden1 / net_input_for_hidden1 = net_output_of_hidden1 * (1 - net_output_of_hidden1);

        (2f) net_input_for_hidden1 / weight2[1] is calculated as follows:
        net_input_for_hidden1 = weight1[1] * input[1] + weight2[1] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden1 wrt weight2[1] = input[2]

        2a1 = 2b * 2c * 2d * 2e * 2f
        var _2a1 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight3[1]) * (net_output_of_hidden1 * (1 - net_output_of_hidden1)) * (input[2]);

        (2a2) = (3b) (total_error_output2 / net_output_of_output2) * (3c) (net_output_of_output2 / net_input_for_output2) * (3d) (net_input_for_output2 / net_output_of_hidden1) * (3e) (net_output_of_hidden1 / net_input_for_hidden1) * (3f) (net_input_for_hidden1 / weight2[1])

        (3b) total_error_output2 / net_output_of_output2 = net_output_of_output2 - target_output2; (EQN 1)
        (3c) net_output_of_output2 / net_input_for_output2 = net_output_of_output2 * (1 - net_output_of_output2) (EQN 2)
        
        (3d) net_input_for_output2 / net_output_of_hidden1 is calculated as follows:
        net_input_for_output2 = net_output_of_hidden1 * weight3[2] + net_output_of_hidden2 * weight4[2] + b2 * 1;
        Partial derivative of net_input_for_output2 wrt net_output_of_hidden1 = weight3[2]

        (3e) net_output_of_hidden1 / net_input_for_hidden1 = net_output_of_hidden1 * (1 - net_output_of_hidden1);

        (3f) net_input_for_hidden1 / weight2[1] is calculated as follows:
        net_input_for_hidden1 = weight1[1] * input[1] + weight2[1] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden2 wrt weight2[1] = input[2]

        2a2 = 3b * 3c * 3d * 3e * 3f
        var _2a2 = (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight3[2]) * (net_output_of_hidden1 * (1 - net_output_of_hidden1)) * (input[2]);

        (q) delta_total_error_wrt_weight2_1 = _2a1 + _2a2;
        delta_total_error_wrt_weight2_1 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight3[1]) * (net_output_of_hidden1 * (1 - net_output_of_hidden1)) * (input[2]) + (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight3[2]) * (net_output_of_hidden1 * (1 - net_output_of_hidden1)) * (input[2]);
    */

    delta_total_error_wrt_weight2_1 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight3[1]) * (net_output_of_hidden1 * (1 - net_output_of_hidden1)) * (input[2]) + (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight3[2]) * (net_output_of_hidden1 * (1 - net_output_of_hidden1)) * (input[2]);
    console.log("q", delta_total_error_wrt_weight2_1);

    new_weight2[1] = weight2[1] - (learning_rate * delta_total_error_wrt_weight2_1);
    console.log({new_weight2});

    /*
        CALCULATE NEW WEIGHT2[2]

        total_error / weight2[2] = (2a1) + (2a2)

        (2a1) = (2b) (total_error_output1 / net_output_of_output1) * (2c) (net_output_of_output1 / net_input_for_output1) * (2d) (net_input_for_output1 / net_output_of_hidden2) * (2e) (net_output_of_hidden2 / net_input_for_hidden2) * (2f) (net_input_for_hidden2 / weight2[2])

        (2b) total_error_output1 / net_output_of_output1 = net_output_of_output1 - target_output1; (EQN 1)
        (2c) net_output_of_output1 / net_input_for_output1 = net_output_of_output1 * (1 - net_output_of_output1) (EQN 2)
        
        (2d) net_input_for_output1 / net_output_of_hidden2 is calculated as follows:
        net_input_for_output1 = net_output_of_hidden1 * weight3[1] + net_output_of_hidden2 * weight4[1] + b2 * 1;
        Partial derivative of net_input_for_output1 wrt net_output_of_hidden2 = weight4[1]

        (2e) net_output_of_hidden2 / net_input_for_hidden2 = net_output_of_hidden2 * (1 - net_output_of_hidden2);

        (2f) net_input_for_hidden2 / weight2[2] is calculated as follows:
        net_input_for_hidden2 = weight1[2] * input[1] + weight2[2] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden2 wrt weight2[2] = input[2]

        2a1 = 2b * 2c * 2d * 2e * 2f
        var _2a1 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight4[1]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[2]);

        (2a2) = (3b) (total_error_output2 / net_output_of_output2) * (3c) (net_output_of_output2 / net_input_for_output2) * (3d) (net_input_for_output2 / net_output_of_hidden2) * (3e) (net_output_of_hidden2 / net_input_for_hidden2) * (3f) (net_input_for_hidden2 / weight2[2])

        (3b) total_error_output2 / net_output_of_output2 = net_output_of_output2 - target_output2; (EQN 1)
        (3c) net_output_of_output2 / net_input_for_output2 = net_output_of_output2 * (1 - net_output_of_output2) (EQN 2)
        
        (3d) net_input_for_output2 / net_output_of_hidden2 is calculated as follows:
        net_input_for_output2 = net_output_of_hidden1 * weight3[2] + net_output_of_hidden2 * weight4[2] + b2 * 1;
        Partial derivative of net_input_for_output2 wrt net_output_of_hidden2 = weight4[2]

        (3e) net_output_of_hidden2 / net_input_for_hidden2 = net_output_of_hidden2 * (1 - net_output_of_hidden2);

        (3f) net_input_for_hidden2 / weight2[2] is calculated as follows:
        net_input_for_hidden2 = weight1[2] * input[1] + weight2[2] * input[2] + b1 * 1;
        Partial derivative of net_input_for_hidden2 wrt weight2[2] = input[2]

        2a2 = 3b * 3c * 3d * 3e * 3f
        var _2a2 = (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight4[2]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[2]);

        (r) delta_total_error_wrt_weight2_2 = _2a1 + _2a2;
        delta_total_error_wrt_weight2_2 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight4[1]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[2]) + (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight4[2]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[2]);
    */

    delta_total_error_wrt_weight2_2 = (net_output_of_output1 - target_output1) * (net_output_of_output1 * (1 - net_output_of_output1)) * (weight4[1]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[2]) + (net_output_of_output2 - target_output2) * (net_output_of_output2 * (1 - net_output_of_output2)) * (weight4[2]) * (net_output_of_hidden2 * (1 - net_output_of_hidden2)) * (input[2]);
    console.log("r", delta_total_error_wrt_weight2_2);

    new_weight2[2] = weight2[2] - (learning_rate * delta_total_error_wrt_weight2_2);
    console.log({new_weight2});

    console.log("old weight1", {weight1});
    console.log("old weight2", {weight2});
    console.log("old weight3", {weight3});
    console.log("old weight4", {weight4});

    weight1 = new_weight1;
    weight2 = new_weight2;
    weight3 = new_weight3;
    weight4 = new_weight4;

    console.log("new weight1", {weight1});
    console.log("new weight2", {weight2});
    console.log("new weight3", {weight3});
    console.log("new weight4", {weight4});
}