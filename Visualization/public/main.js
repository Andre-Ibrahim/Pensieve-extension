const S_INFO = 6;
const A_DIM = 9;
let factory = dashjs.FactoryMaker;
// Initialize dash.js player
const player = dashjs.MediaPlayer().create();
player.initialize(document.querySelector("#videoPlayer1"), "https://bitmovin-a.akamaihd.net/content/sintel/sintel.mpd", true);

const player2 = dashjs.MediaPlayer().create();
player2.initialize(document.querySelector("#videoPlayer2"), "https://bitmovin-a.akamaihd.net/content/sintel/sintel.mpd", true);

const throughputChartCtx = document.getElementById('throughputChart1');
const bitrateChartCtx = document.getElementById('bitrateChart1');

const throughputChartCtx2 = document.getElementById('throughputChart2');
const bitrateChartCtx2 = document.getElementById('bitrateChart2');

const throughputChart = new Chart(throughputChartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Throughput (kbps)',
            data: [],
            borderColor: 'blue',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { title: { display: true, text: 'Throughput (kbps)' } }
        }
    }
});

const throughputChart2 = new Chart(throughputChartCtx2, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Throughput (kbps)',
            data: [],
            borderColor: 'blue',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { title: { display: true, text: 'Throughput (kbps)' } }
        }
    }
});

const bitrateChart = new Chart(bitrateChartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Selected Bitrate',
            data: [],
            borderColor: 'green',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: {
                title: {
                    display: true,
                    text: 'Bitrate level'
                },
                min: 0, // Optional, sets the minimum value
                max: 8  // Sets the maximum value of the y-axis
            }
        }
    }
});

const bitrateChart2 = new Chart(bitrateChartCtx2, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Selected Bitrate',
            data: [],
            borderColor: 'green',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: {
                title: {
                    display: true,
                    text: 'Bitrate level'
                },
                min: 0,
                max: 8 
            }
        }
    }
});

let model;
let oldModel;

totalNumberOfChunks = 1400;
chunksremaining = totalNumberOfChunks;
chunksremaining2 = totalNumberOfChunks;
counter = 0;
counter2 = 0;
// Load TensorFlow.js model once
async function loadModel() {
    model = await tf.loadGraphModel('./onnx_model_js/heterogenous_reward5_exp2_800epochs/model.json');
    console.log("Actor model loaded successfully");
}

async function loadOldModel() {
    oldModel = await tf.loadGraphModel('./onnx_model_js/heterogenous_retrained_new_video/model.json');
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

player.updateSettings({
    streaming: {
        delay: {
            liveDelay: 4
        },
        liveCatchup: {
            maxDrift: 0,
            playbackRate: {
                max: 1,
                min: -0.5
            }
        }
    }
});


let DashManifestModel = factory.getSingletonFactoryByName('DashManifestModel');
let dashManifest = DashManifestModel(this.context).getInstance();

let bandwidth = 0;

// Persistent state array (keeps values between predictions)
const state = Array.from({ length: S_INFO }, () => Array(A_DIM).fill(0));

currentBandwidth = 50;

player.on(dashjs.MediaPlayer.events.MANIFEST_LOADED, function (e) {
    console.log("Manifest loaded");
    console.log(e);
})

// Event listener for Dash.js
player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, async function (event) {

    if(event.mediaType !== "video") return;

    dashMetrics = player.getDashMetrics();

    if (!model) {
        await loadModel();
        if (!model) return;
    }

    if (!oldModel) {
        await loadOldModel();
        if (!oldModel) return;
    }

    const metrics = player.getDashMetrics();
    const bufferLevel = dashMetrics.getCurrentBufferLevel("video", true);
    
    const bitrateList = player.getBitrateInfoListFor("video");

    const streamInfo = player.getActiveStream().getStreamInfo();
    const periodIdx = streamInfo.index;
    
    const dashAdapter = player.getDashAdapter();
    const videoRepresentations = dashAdapter.getAdaptationForType(periodIdx, 'video').Representation;

    const segmentSizeEstimates = [];

    videoRepresentations.forEach(rep => {
        const bitrate = rep.bandwidth; // in bits per second
        const segmentTemplate = rep.SegmentTemplate || videoAdaptation.SegmentTemplate;
        const segmentDuration = segmentTemplate.duration / segmentTemplate.timescale; // in seconds
        const estimatedSizeBytes = (bitrate / 8) * segmentDuration; // Convert bits to bytes
        segmentSizeEstimates.push(estimatedSizeBytes.toFixed(2) / 1000);
    });

    console.log(segmentSizeEstimates);

    const previousQuality = player.getQualityFor("video");

    let httpRequest = dashMetrics.getCurrentHttpRequest("video", true);

    lengthcontent = httpRequest._responseHeaders.match(/content-length:\s*(\d+)/i);

    // Update state values without re-initializing
    const normalizedBitrates = bitrateList.map(b => b.bitrate / 1000.0);

    // Ensure normalizedBitrates has exactly A_DIM values
    while (normalizedBitrates.length < A_DIM) normalizedBitrates.push(0);
    normalizedBitrates.length = A_DIM;

    console.log(normalizedBitrates);

    for (let i = 0; i < S_INFO; i++) {
        for (let j = 1; j < A_DIM; j++) {
            state[i][j - 1] = state[i][j];
        }
    }

    rebufferTime = 0;
    
    // ms
    delay = event.request.bytesLoaded / currentBandwidth;

    if (bufferLevel < delay * 1000){
        rebufferTime = delay * 1000 - bufferLevel;
        rebufferTime = rebufferTime / 1000;
    }

    state[0][A_DIM - 1] = (bitrateList[previousQuality].bitrate / 1000) / normalizedBitrates[A_DIM - 1];
    state[1][A_DIM - 1] = bufferLevel / 10.0;
    state[2][A_DIM - 1] = currentBandwidth / 1000;
    state[3][A_DIM - 1] = delay + rebufferTime;
    state[4] = segmentSizeEstimates;
    state[5][A_DIM - 1] = chunksremaining / totalNumberOfChunks;

    console.log(event.request.bytesLoaded / currentBandwidth);

    const stateTensor = tf.tensor(state).expandDims(0);

    const prediction = await model.executeAsync(stateTensor);

    const actionProb = await prediction.data();

    const actionProbTensor = tf.tensor(actionProb);
    const qualityIndex = actionProbTensor.argMax().dataSync()[0];

    console.log("Predicted quality index:", qualityIndex);

    player.setQualityFor("video", qualityIndex);

    throughputChart.data.labels.push(counter);
    throughputChart.data.datasets[0].data.push(currentBandwidth);
    throughputChart.update();

    bitrateChart.data.labels.push(counter);
    bitrateChart.data.datasets[0].data.push(qualityIndex);
    bitrateChart.update();

    counter++;
    chunksremaining -= 1;
});


convertQualityLevels = (qualityLevels) => {
    return qualityLevels.map(quality => quality.bitrate / 1000);
}

const NETWORK_BANDWIDTH = {
    "1G": 50,
    "2G": 150,
    "2G+": 300,
    "2G++": 450,
    "3G": 1500,
    "4G": 5000,
    "5G": 8000,
};

function addNoiseToBandwidth(bandwidth, mean = 0, stdDev = 5) {
    // Function to generate random normal distribution noise
    function randomNormal(mean, stdDev) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random(); // Ensure u is not zero
        while (v === 0) v = Math.random(); // Ensure v is not zero
        let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z * stdDev + mean;
    }

    // Add noise to the original bandwidth value
    const noise = randomNormal(mean, stdDev);
    return bandwidth + noise;
}


// List of network types
const networkTypes = ["1G", "2G", "2G+","2G++","3G", "4G", "5G"];

// Function to pick a random network type
function pickRandomNetworkType() {
    const randomIndex = Math.floor(Math.random() * networkTypes.length);
    return networkTypes[randomIndex];
}

function changeNetworkConditions() {
    const currentNetwork = pickRandomNetworkType();
    bandwidth = NETWORK_BANDWIDTH[currentNetwork];
    currentBandwidth = addNoiseToBandwidth(bandwidth, 0 , 100);

    if(currentBandwidth < 0) currentBandwidth = 0;

    console.log(`Switching to ${currentNetwork} with bandwidth: ${currentBandwidth} kbps`);

    player.updateSettings({
        streaming: {
            abr: {
                maxBitrate: { video: currentBandwidth }
            }
        }
    });
}

setInterval(changeNetworkConditions, 2000);

player2.updateSettings({
    streaming: {
        buffer: {
            stableBufferTime: 8,
            bufferTimeAtTopQuality: 8,
            bufferTimeAtTopQualityLongForm: 8
        }
    }
});

player2.updateSettings({
    streaming: {
        delay: {
            liveDelay: 4
        },
        liveCatchup: {
            maxDrift: 0,
            playbackRate: {
                max: 1,
                min: -0.5
            }
        }
    }
});

player2.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, async function (event) {

    if(event.mediaType !== "video") return;

    dashMetrics = player.getDashMetrics();

    if (!oldModel) {
        await loadOldModel();
        if (!oldModel) return;
    }
    const bufferLevel = dashMetrics.getCurrentBufferLevel("video", true);
    
    const bitrateList = player.getBitrateInfoListFor("video");

    const streamInfo = player.getActiveStream().getStreamInfo();
    const periodIdx = streamInfo.index;
    
    const dashAdapter = player.getDashAdapter();
    const videoRepresentations = dashAdapter.getAdaptationForType(periodIdx, 'video').Representation;

    const segmentSizeEstimates = [];

    videoRepresentations.forEach(rep => {
        const bitrate = rep.bandwidth; // in bits per second
        const segmentTemplate = rep.SegmentTemplate || videoAdaptation.SegmentTemplate;
        const segmentDuration = segmentTemplate.duration / segmentTemplate.timescale; // in seconds
        const estimatedSizeBytes = (bitrate / 8) * segmentDuration; // Convert bits to bytes
        segmentSizeEstimates.push(estimatedSizeBytes.toFixed(2) / 1000);
    });

    console.log(segmentSizeEstimates);

    const previousQuality = player.getQualityFor("video");

    let httpRequest = dashMetrics.getCurrentHttpRequest("video", true);

    lengthcontent = httpRequest._responseHeaders.match(/content-length:\s*(\d+)/i);

    const normalizedBitrates = bitrateList.map(b => b.bitrate / 1000.0);

    while (normalizedBitrates.length < A_DIM) normalizedBitrates.push(0);
    normalizedBitrates.length = A_DIM;

    console.log(normalizedBitrates);

    for (let i = 0; i < S_INFO; i++) {
        for (let j = 1; j < A_DIM; j++) {
            state[i][j - 1] = state[i][j];
        }
    }

    rebufferTime = 0;
    
    // ms
    delay = event.request.bytesLoaded / currentBandwidth;

    if (bufferLevel < delay * 1000){
        rebufferTime = delay * 1000 - bufferLevel;
        rebufferTime = rebufferTime / 1000;
    }

    state[0][A_DIM - 1] = (bitrateList[previousQuality].bitrate / 1000) / normalizedBitrates[A_DIM - 1];
    state[1][A_DIM - 1] = bufferLevel / 10.0;
    state[2][A_DIM - 1] = currentBandwidth / 1000;
    state[3][A_DIM - 1] = delay + rebufferTime;
    state[4] = segmentSizeEstimates;
    state[5][A_DIM - 1] = chunksremaining / totalNumberOfChunks;

    console.log(event.request.bytesLoaded / currentBandwidth);

    const stateTensor = tf.tensor(state).expandDims(0);

    const prediction = await oldModel.executeAsync(stateTensor);

    const actionProb = await prediction.data();

    const actionProbTensor = tf.tensor(actionProb);
    const qualityIndex = actionProbTensor.argMax().dataSync()[0];

    console.log("Predicted quality index:", qualityIndex);

    player2.setQualityFor("video", qualityIndex);

    bitrateChart2.data.labels.push(counter2);
    bitrateChart2.data.datasets[0].data.push(qualityIndex);
    bitrateChart2.update();

    throughputChart2.data.labels.push(counter);
    throughputChart2.data.datasets[0].data.push(currentBandwidth);
    throughputChart2.update();

    counter2++;
    chunksremaining2 -= 1;
});