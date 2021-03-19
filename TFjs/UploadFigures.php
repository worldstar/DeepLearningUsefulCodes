<!-- <!DOCTYPE html> -->
<!-- release v5.1.1, copyright 2014 - 2020 Kartik Visweswaran -->
<!--suppress JSUnresolvedLibraryURL -->
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>Tensorflow</title>
    
    <link rel="stylesheet" href="/vendor/bootstrap-4.5.0-dist/css/bootstrap.min.css" crossorigin="anonymous">
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" crossorigin="anonymous">
    <script src="/vendor/bootstrap-4.5.0-dist/js/jquery-3.3.1.min.js" crossorigin="anonymous"></script>
    <script src="/vendor/bootstrap-4.5.0-dist/js/bootstrap.bundle.min.js" crossorigin="anonymous"></script>
 <\>
<body>
  <!-- <h3><input id="upload-input" type="file" multiple="multiple"accept="image/gif, image/jpg, image/png"/></h3> -->
  <div class="custom-file">
  <input type="file" class="custom-file-input" id="upload-input" multiple="multiple" accept="image/gif, image/jpg, image/png">
  <label class="custom-file-label" for="customFile">Choose file</label>
  </div>
 <table class="table table-dark table-striped">
      <tr>
        <th>圖片</th>
        <th>參數</th>
      </tr>
    <tbody id="upload">

    </tbody>
  </table>

 </body>
<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-automl"></script>
<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>

<script>
$('#upload-input').change(async function(e){

        var files = $("#upload-input")[0].files;

        
        for (var i = 0; i < files.length; i++)
        {
          var url = window.URL.createObjectURL(eval(files[i]));
          // alert(url);
          // $("#upload").attr('src', url);
           $("#upload").append(` <tr ><td width="30%"><img id='salad${i}' src=${url}></td width="70%"><td id='sal${i}'>預測中...</td></tr>`);
        }
        var str = '';
        var predictValue = new Array();
        var model = await tf.automl.loadObjectDetection('model.json');
        var options = {score: 0.5, iou: 0.5, topk: 20};
        var time=0;
        for (var i = 0; i < files.length; i++)
        {
          

           var img = document.getElementById(`salad${i}`);
           var predictions = await model.detect(img, options);

           console.log(predictions);
           if(predictions.length == 0){
            console.log('not found!!');
            str = 'not found!!\n';
            document.getElementById('sal'+i).innerHTML='not found!!\n';
          }else{
            var myJSON1 = JSON.stringify(predictions[0]["box"]["left"]);
            var myJSON11 = JSON.stringify(predictions[0]["box"]["top"]);
            var myJSON12 = JSON.stringify(predictions[0]["box"]["width"]);
            var myJSON13 = JSON.stringify(predictions[0]["box"]["height"]);
            var myJSON2 = JSON.stringify(predictions[0]["label"]);
            var myJSON3 = JSON.stringify(predictions[0]["score"]);
            // str = myJSON1.toString() + "," + myJSON11.toString() + "," + myJSON12.toString() + "," + 
            //     myJSON13.toString() + "," + myJSON2.toString() + "," + myJSON3.toString() + "," + fname + "\n";
            document.getElementById('sal'+i).innerHTML ="left:"+myJSON1+"<br> top:"+myJSON11+"<br> width:"+myJSON12+"<br> height:"+myJSON13+"<br> label:"+myJSON2+"<br> score:"+myJSON3;
          }

          predictValue.push(str);
        }
        
});
// php -S localhost:8000

</script> 
</html>

