var fs = require("fs");
let results = JSON.parse(fs.readFileSync("./finalBadResults.json"));

let k = 0;
//Find the original snapshots where the recognition was made

let newResults = [];

results.forEach(res => {
    res.faceRecognitions.forEach(faceRec => {
        newResults.push({
            snapId: res._id.$oid,
            userQuality: faceRec.UserQuality,
            bottom: faceRec.boundingBox.bottom,
            top: faceRec.boundingBox.top,
            left: faceRec.boundingBox.left,
            right: faceRec.boundingBox.right,
            confidence: faceRec.confidence,
            distance: faceRec.distance,
            faceId: faceRec.faceId,
            recogType: faceRec.recogType,
            recognitions: faceRec.recognitions
        });
    });
});

var fs = require("fs");

//var json = JSON.parse(fs.readFileSync('./badResults2.json', 'utf-8'));
var json = newResults;

var csvWriter = require('csv-write-stream');
var writer = csvWriter({headers: ["Number", "SnapId", "Distance", "Bad Matches", "Total Matches", "Confidence", "FaceId", "Top", "Bottom", "Left", "Right", "UserEstimate"]});

writer.pipe(fs.createWriteStream("./snap_scores.csv"));

let i = 0;
let recogCount = 0;
let newCount = 0;
let bad = {};

json.forEach(snap => {
    i++;
    let tempRes = [];
    tempRes = tempRes.concat(`${snap.confidence}`);

    if(snap.recogType == "recognition") {
        recogCount++;
        let result = [];
        let j=0;
        let badCount = 0;
        let wroteRes = false;
        let wroteTop = false;
        result = result.concat([`${i}`, `${snap.snapId}`]);
        snap.recognitions.forEach(recog => {
            j++;

            if(recog.hasOwnProperty("badQuality") && recog.badQuality) {
                bad[`${recog.faceId}`] = true;
                badCount++;
            }

            //if(recog.hasOwnProperty("humanCorrect") && recog.humanCorrect == true) {
                //if(j == 1) {
                    //if(wroteRes == false){
                        //result = result.concat([`TP`]);
                        //wroteRes = true;
                    //}
                    //if(wroteTop == false){
                        //result = result.concat([`T`, `${recog.distance}`]);
                        //wroteTop = true;
                    //}
                //} else if (j <= 5) {
                    //if(wroteRes == false){
                        //result = result.concat([`FP`]);
                        //wroteRes = true;
                    //}
                    //if(wroteTop == false){
                        //result = result.concat([`T`, `${recog.distance}`]);
                        //wroteTop = true;
                    //}
                //} else {
                    //if(wroteRes == false){
                        //result = result.concat([`FP`]);
                        //wroteRes = true;
                    //}
                //}
            //} else {
                //if(wroteRes == false){
                    //result = result.concat([`FP`]);
                    //wroteRes = true;
                //}
                //if(wroteTop == false){
                    //result = result.concat([`F`, `${recog.distance}`]);
                    //wroteTop = true;
                //}
            //}
        });

        result.push(snap.distance);
        result.push(badCount);
        result.push(j);
        result = result.concat(tempRes)
        result = result.concat([`${snap.faceId}`, `${snap.top}`, `${snap.bottom}`, `${snap.left}`, `${snap.right}`, `${snap.userQuality}`]);
        writer.write(result);
    } else {
        newCount++;
        //let result =[`${i}`, `${snap.snapId}`, `New`, `New`, `New`, `New`, `New`];
        let result =[`${i}`, `${snap.snapId}, 'New', 'New', 'New'`];
        result = result.concat(tempRes);
        result = result.concat([`${snap.faceId}`, `${snap.top}`, `${snap.bottom}`, `${snap.left}`, `${snap.right}`, `${snap.userQuality}`]);
        writer.write(result);
    }
});

writer.end();

//fs.writeFileSync("./wikus1.json", JSON.stringify(newResults, null, 4));
