<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script src="https://unpkg.com/@tensorflow/tfjs-automl"></script>

<script>
async function run() {
  document.write(`<img id='salad' name='salad' crossorigin='anonymous' src='./photo/1546fea49902ee2dbc640573d57a6d1d-_1_1-00021.png'>`);
  const model = await tf.automl.loadObjectDetection('model.json');
  const img = document.getElementById('salad');
  const options = {score: 0.5, iou: 0.5, topk: 20};
  const predictions = await model.detect(img, options);
  console.log(predictions);
  // Show the resulting object on the page.
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(predictions, null, 2);
  document.body.append(pre);
}
run();
</script>