if (window.File && window.FileReader && window.FileList && window.Blob){
    // alle File Apis vorhanden
}else {
    alert("no File APIs");
}

let mim = new mimir(),
    bow = mim.exports.bow,
    dict = mim.exports.dict;
let voc = dict("ich bin blöd");
console.log(bow("blöd", voc));



// let fs = require('fs');
// let brain = require('brain.js');
// let mimir = require('./mimirIndex'),
//   bow = mimir.bow,
//   dict = mimir.dict;
// let trumpTrainData = JSON.parse(
//   fs.readFileSync('./Rohdaten/trumpTrainingData.json')
// );
// let trumpTestData = JSON.parse(
//   fs.readFileSync('./Rohdaten/trumpTestData.json')
// );
// let kimTrainData = JSON.parse(
//   fs.readFileSync('./Rohdaten/kardashianTrainingData.json')
// );
// let kimTestData = JSON.parse(
//   fs.readFileSync('./Rohdaten/kardashianTestData.json')
// );




// const generateTestData = (dataArr, namesArr) => {
//   let trainDat = [];
//   dataArr.forEach((element, index) => {
//     element.forEach(element => {
//       trainDat.push({
//         input: bow(element, voc),
//         output: { [namesArr[index]]: 1 }
//       });
//     });
//   });
//   return trainDat;
// };

// const cleanData = data => {
//   data.map((element, index) => {
//     if (element.indexOf('"@') >= 0) {
//       data.splice(index, 1);
//     }
//   });
// };

// cleanData(trumpTrainData);
// cleanData(trumpTestData);
// cleanData(kimTrainData);
// cleanData(kimTestData);

// const allRawTrainData = trumpTrainData.concat(
//   kimTrainData,
//   trumpTestData,
//   kimTestData
// );
// const voc = dict(allRawTrainData);
// const generatedTrianigData = generateTestData(
//   [trumpTrainData, kimTrainData],
//   ['Trump', 'Kim']
// );

// const net = new brain.NeuralNetwork({ hiddenLayers: [1000] });
// net.setActivation('sigmoid');
// const MAXITERATIONS = 100;
// const conf = {
//   iterations: MAXITERATIONS,
//   log: true,
//   logPeriod: 10,
//   /* callback: () => {
//     for (var i = generatedTrianigData.length - 1; i > 0; i--) {
//       var j = Math.floor(Math.random() * (i + 1));
//       var temp = generatedTrianigData[i];
//       generatedTrianigData[i] = generatedTrianigData[j];
//       generatedTrianigData[j] = temp;
//     }
//   },
//   callbackPerid: 10,*/
//   //learningRate: 0.3,
//   errorThresh: 0.000005
// };


// const trainAndSave = () => {
//   net.train(generatedTrianigData, conf);
//   let lastJSON = net.toJSON();
//   fs.writeFileSync('./JSON/lastJSON.json', JSON.stringify(lastJSON));
// };

// const testTraining = (versuche) => {
//   let testResult = '';
//   let fromTrump = true;
//   let cntRichtig = versuche;
//   for (let i = 0; i < versuche; i++) {
//     if (Math.random() < 0.5) {
//       fromTrump = true;
//       testResult =
//         trumpTestData[Math.round(Math.random() * trumpTestData.length)];
//       console.log('Trump', testResult.trimRight());
//     } else {
//       fromTrump = false;
//       testResult = kimTestData[Math.round(Math.random() * kimTestData.length)];
//       console.log('Kim', testResult.trimRight());
//     }
//     let result = net.run(bow(testResult, voc));
//     console.log('Result: ', result);
//     if (fromTrump === true && result.Trump > result.Kim) {
//       console.log('RICHTIG es war TRUMP');
//     } else if (fromTrump === false && result.Trump < result.Kim) {
//       console.log('RICHTIG es war KIM');
//     } else {
//       console.log('FALSCH');
//       cntRichtig--;
//     }

//     console.log('----------------------------------------');
//   }
//   let prozentRichtig = cntRichtig/versuche*100;
//   console.log('+++++++++++++++++++ ' +prozentRichtig+ ' % ++++++++++++++++++++');
// };
// trainAndSave();
// testTraining(100);