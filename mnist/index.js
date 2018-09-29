let digits = Array(10);
const height = 28;
const width = 28;
let singleDigits = [];
let trainDat = [];

const conf = {

    iterations: 50,
    log: true,
    learningRate: 0.1,
    errorThresh: 0.001
};
const net = new brain.NeuralNetwork({ hiddenLayers: [10] });
net.setActivation('sigmoid');

process();

async function process() {
    for (let i = 0; i < 10; i++) {
        let d = await $.getJSON("./digits_as_json/" + i + ".json");
        digits.push(d.data);
    }
    //showAll();
    createSingleDigits();
    buildTrainData();
    shuffleArray(trainDat);
    train().then(data => {
        console.log(data);
        testDraw();
    });

}

function testDraw() {
    for (let i = 0; i < 5; i++) {
        let tmpIMG = trainDat[i * 1100].input;
        drawSingleDigit(tmpIMG);
        drawSingleDigit(net.run(tmpIMG));
    }
}

function createSingleDigits() {

    digits.forEach(data => {
        let index = 0;
        while (index < data.length) {
            let buffer = [];
            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    var pos = (y * width + x) * 1; // position in buffer based on x and y
                    var posInDigitArray = index + y * width + x;
                    buffer[pos] = data[posInDigitArray];           // some R value [0, 255]
                    // buffer[pos + 1] = data[posInDigitArray];           // some G value
                    // buffer[pos + 2] = data[posInDigitArray];           // some B value
                    // buffer[pos + 3] = 1;           // set alpha channel
                }
            }

            singleDigits.push(buffer);
            //console.log(buffer.length, data.length);
            index += height * width;
        }
    });
}

function showAll() {
    digits.forEach(data => {
        let index = 0;
        while (index < data.length) {
            let buffer = new Uint8ClampedArray(height * width * 4);
            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    var pos = (y * width + x) * 4; // position in buffer based on x and y
                    var posInDigitArray = index + y * width + x;
                    buffer[pos] = data[posInDigitArray] * 255;           // some R value [0, 255]
                    buffer[pos + 1] = data[posInDigitArray] * 255;           // some G value
                    buffer[pos + 2] = data[posInDigitArray] * 255;           // some B value
                    buffer[pos + 3] = 255;           // set alpha channel
                }
            }
            const canvas = document.createElement("canvas");
            canvas.height = height;
            canvas.width = width;
            const ctx = canvas.getContext("2d");
            var idata = ctx.createImageData(width, height);
            idata.data.set(buffer);
            ctx.putImageData(idata, 0, 0);
            output.appendChild(canvas);
            index += height * width;
        }
    });
}



function buildTrainData() {
    trainDat = [];
    let index = 0;
    singleDigits.forEach(digit => {
        let
        let data = {
            input: digit,
            output:
        };
        trainDat.push(data);
    });
}

async function train() {
    return net.trainAsync(trainDat, conf);
}


function drawSingleDigit(image) {
    let buffer = new Uint8ClampedArray(height * width * 4);
    for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4; // position in buffer based on x and y
            var posInDigitArray = 0 + y * width + x;
            buffer[pos] = image[posInDigitArray] * 255;           // some R value [0, 255]
            buffer[pos + 1] = image[posInDigitArray] * 255;           // some G value
            buffer[pos + 2] = image[posInDigitArray] * 255;           // some B value
            buffer[pos + 3] = 255;           // set alpha channel
        }
    }
    const canvas = document.createElement("canvas");
    canvas.height = height;
    canvas.width = width;
    const ctx = canvas.getContext("2d");
    var idata = ctx.createImageData(width, height);
    idata.data.set(buffer);
    ctx.putImageData(idata, 0, 0);
    output.appendChild(canvas);
}

function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}