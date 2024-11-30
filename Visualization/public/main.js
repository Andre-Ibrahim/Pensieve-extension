const S_INFO = 6;
const A_DIM = 9;
let factory = dashjs.FactoryMaker;
// Initialize dash.js player
const player = dashjs.MediaPlayer().create();
player.initialize(document.querySelector("#videoPlayer"), "https://dash.akamaized.net/envivio/EnvivioDash3/manifest.mpd", true);

let model;

// Load TensorFlow.js model once
async function loadModel() {
    model = await tf.loadGraphModel('./onnx_model_js/heterogenous_switch_rate_tuned/model.json');
    console.log("Actor model loaded successfully");
}

// Set buffer settings for Dash.js
player.updateSettings({
    streaming: {
        buffer: {
            stableBufferTime: 8,
            bufferTimeAtTopQuality: 8,
            bufferTimeAtTopQualityLongForm: 8
        }
    }
});

let DashManifestModel = factory.getSingletonFactoryByName('DashManifestModel');
let dashManifest = DashManifestModel(this.context).getInstance();

let bandwidth = 0;

// Persistent state array (keeps values between predictions)
const state = Array.from({ length: S_INFO }, () => Array(A_DIM).fill(0));

// Event listener for Dash.js
player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, async function () {
    console.log("Fragment loading completed");

    dashMetrics = player.getDashMetrics();

    // Load the model (ensure it's loaded only once)
    if (!model) {
        await loadModel();
        if (!model) return; // Exit if model failed to load
    }

    const metrics = player.getDashMetrics();
    const bufferLevel = dashMetrics.getCurrentBufferLevel("video", true);; // Fallback if undefined
    console.log("Buffer level:", bufferLevel);
    
    const bitrateList = player.getBitrateInfoListFor("video");

    const throughput =  3000; // Fallback if undefined
    const previousQuality = player.getQualityFor("video");

    let httpRequest = dashMetrics.getCurrentHttpRequest("video", true);


    console.log("HTTP bandwidth:", httpRequest);

    
    const downloadTimesec = httpRequest.tresponse.getSeconds() - httpRequest.trequest.getSeconds();
    const downloadTime = httpRequest.tresponse.getMilliseconds() - httpRequest.trequest.getMilliseconds();



    lengthcontent = httpRequest._responseHeaders.match(/content-length:\s*(\d+)/i);

    
    console.log("Download time:", downloadTime);

    console.log("Content length:", parseInt(lengthcontent[1], 10));


    // Update state values without re-initializing
    const normalizedBitrates = bitrateList.map(b => b.bitrate / 1000.0);

    console.log("Normalized bitrates:", normalizedBitrates);

    // Ensure normalizedBitrates has exactly A_DIM values
    while (normalizedBitrates.length < A_DIM) normalizedBitrates.push(0);
    normalizedBitrates.length = A_DIM;

    for (let i = 0; i < S_INFO; i++) {
        for (let j = 1; j < A_DIM; j++) {
            state[i][j - 1] = state[i][j];
        }
    }

    state[0][A_DIM - 1] = (bitrateList[previousQuality].bitrate / 1000) / normalizedBitrates[A_DIM - 1] ; // Normalized last quality
    state[1][A_DIM - 1] = bufferLevel / 10.0;
    state[2][A_DIM - 1] = bandwidth;
    state[3][A_DIM - 1] = 40 / 1000.0;
    state[4] = normalizedBitrates; // Replace the entire row
    state[5][A_DIM - 1] = 10 / bitrateList.length;

    console.log("Updated state:", state);

    const stateTensor = tf.tensor(state).expandDims(0);

    const prediction = await model.executeAsync(stateTensor);

    const actionProb = await prediction.data();

    const actionProbTensor = tf.tensor(actionProb);
    const qualityIndex = actionProbTensor.argMax().dataSync()[0];

    console.log("Predicted quality index:", qualityIndex);

    player.setQualityFor("video", qualityIndex);
});


convertQualityLevels = (qualityLevels) => {
    return qualityLevels.map(quality => quality.bitrate / 1000);
}



const NETWORK_BANDWIDTH = {
    "3G": 1000,
    "4G": 5000,
    "5G": 20000,
};

// List of network types
const networkTypes = ["3G", "4G", "5G"];

// Function to pick a random network type
function pickRandomNetworkType() {
    const randomIndex = Math.floor(Math.random() * networkTypes.length);
    return networkTypes[randomIndex];
}

// Function to update network conditions randomly
function changeNetworkConditions() {
    const currentNetwork = pickRandomNetworkType();
    bandwidth = NETWORK_BANDWIDTH[currentNetwork];
    console.log(`Switching to ${currentNetwork} with bandwidth: ${bandwidth} kbps`);

    // Override ABR rules to simulate network bandwidth
    player.updateSettings({
        streaming: {
            abr: {
                useDefaultABRRules: false,
                maxBitrate: bandwidth
            }
        }
    });
}

setInterval(changeNetworkConditions, 5000);