<?php

$fp = fopen ( "a.txt" , 'w' );
$length = $_POST['length'];
$predictValue = $_POST['predictValue'];
for($i = 0; $i < $length; $i++){
	fwrite($fp, $predictValue[$i]);
}

echo "1" ;
// $myJSON1 = $_POST['myJSON1'];
// $myJSON11 = $_POST['myJSON11'];
// $myJSON12 = $_POST['myJSON12'];
// $myJSON13 = $_POST['myJSON13'];
// $myJSON2 = $_POST['myJSON2'];
// $myJSON3 = $_POST['myJSON3'];
// $filename = $_POST['filename'];


// $txt = $myJSON1.",".$myJSON11.",".$myJSON12.",".$myJSON13.",".$myJSON2.",".$myJSON3.",".$filename."\n";

// $fp = fopen ( "a.txt" , 'a+' );
// fwrite($fp,$txt);

// echo "1" ;
?>
