<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-automl"></script>
<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>

<script>

async function run() {
	
	var jsArray = `<?php $dir="./photo/";
	$file=glob('./photo/*.png'); 
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
	var model = await tf.automl.loadObjectDetection('model.json');
	var options = {score: 0.5, iou: 0.5, topk: 20};
	var time=0;
	for(var i = 0; i < a.length; i++){
		console.log(i+1);
		var ii=i+1;
		// if((i % 50) == 0){
		// 	console.log(i);
		// 	model = '';
		// 	model = await tf.automl.loadObjectDetection('model.json');
		// }
		
		var fname = a[i].split("/")[2];
		var img = document.getElementById(`salad${i}`);

		var start_time = new Date().getTime();
		var predictions = await model.detect(img, options);
		var end_time = new Date().getTime();
		var AllEnd=(end_time - start_time) / 1000 ;
		time+=AllEnd;
		
		
		console.log(AllEnd+"ASEC");
		console.log(time+"SEC");
		console.log(predictions);
		
		var str = '';
		if(predictions.length == 0){
			console.log('not found!!');
			str = fname + '：' + 'not found!!\n';
		}else{
			var myJSON1 = JSON.stringify(predictions[0]["box"]["left"]);
			var myJSON11 = JSON.stringify(predictions[0]["box"]["top"]);
			var myJSON12 = JSON.stringify(predictions[0]["box"]["width"]);
			var myJSON13 = JSON.stringify(predictions[0]["box"]["height"]);
			var myJSON2 = JSON.stringify(predictions[0]["label"]);
			var myJSON3 = JSON.stringify(predictions[0]["score"]);
			str = myJSON1.toString() + "," + myJSON11.toString() + "," + myJSON12.toString() + "," + 
					myJSON13.toString() + "," + myJSON2.toString() + "," + myJSON3.toString() + "," + fname + "\n";
		}

		predictValue.push(str);
	}

	var s=time/ii;
	console.log(s);
	
	$.ajax({
	    type: "POST",
	    url: "message.php",
	    data: {length: predictValue.length, predictValue: predictValue},
	    // traditional:true,
	    async: false,
	    cache: false
	}).done(function(msg) {
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