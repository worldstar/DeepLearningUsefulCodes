

<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>Tensorflow</title>
</head>
 
<body>

</body>

<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-automl"></script>
<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>

<script>

async function run() {
	
	var jsArray = `<?php $dir="./googleAutoML_testimg20210429/test/JPEGImages20210429/";
	$file=glob('./googleAutoML_testimg20210429/test/JPEGImages20210429/*.png'); 
	$filec = count($file);
	$f = array();
	for ($i=0;$i<$filec;$i++){
	array_push($f,$file[$i]);	
	} 
	echo join("|",$f);
	?>`;

	var a = new Array();
	var a = jsArray.split("|");
	var fname = new Array();

	// 在執行document.write() <-- 更新網頁內容時，執行Tensorflow Model 會出現錯誤，因此先更新完畢再執行。
	for(var i = 0; i < a.length; i++){
		document.write(`<img id='salad${i}' name='salad${i}' crossorigin='anonymous' src=${a[i]}>`);
	}
	var predictValue = new Array();
	var model = await tf.automl.loadObjectDetection('./GoogleAutoMLModel20210421/model.json');
	var options = {score: 0.5, iou: 0.5, topk: 20};
	var time=0;
	// var str = '';
	for(var i = 0; i < a.length; i++){
		console.log(i+1);
		var ii=i+1;
		// if((i % 50) == 0){
		// 	console.log(i);
		// 	model = '';
		// 	model = await tf.automl.loadObjectDetection('model.json');
		// }
		
		var fname2 = a[i].split("/")[4];
		var fname3 = fname2.split(".png")[0];
		var img = document.getElementById(`salad${i}`);

		var start_time = new Date().getTime();
		var predictions = await model.detect(img, options);
		var end_time = new Date().getTime();
		var AllEnd=(end_time - start_time) / 1000 ;
		time+=AllEnd;
		
		// console.log(AllEnd+"全部秒數");
		// console.log(time+"當前秒數");
		console.log(predictions);
		
		
		
		if(predictions.length == 0){
			console.log('not found!!');
			str = fname + '：' + 'not found!!\n';
		}else{
			predictValue.push(predictions);
			var str = str +' '+fname3;
			// var myJSON1 = JSON.stringify(predictions[0]["box"]["left"]);
		}
		
	}
	// console.log(predictValue.length+"--");
	// 	predictValue.push(predictions);
	

	var resultarray=[];

	for(var x = 0; x < predictValue.length; x++)
	{
		// console.log(x+"--");
		for(var j = 0; j < predictValue[x].length; j++)
		{

		var test = predictValue[x][j]["label"]+" "+predictValue[x][j]["box"]["left"]+" "+predictValue[x][j]["box"]["top"]+" "+predictValue[x][j]["box"]["width"]+" "+predictValue[x][j]["box"]["height"] + "---" + (x+1);
		// console.log(test);
		resultarray.push(test);

		}

	}
	// console.log(resultarray);
	// var s=time/ii;

	// console.log('平均每張使用秒數：'+s);
	
	$.ajax({
	    type: "POST",
	    url: "message.php",
	    data: {resultarray: resultarray,str:str},
	    // traditional:true,
	    async: false,
	    cache: false
	}).done(function(msg) {
		// console.log(msg);
	    if(msg="0"){
	      console.log("Y");
	    }
	    else{
	      alert("N");
	    }
	});
	

}
run();
// php -S localhost:8000

</script>	
</html>