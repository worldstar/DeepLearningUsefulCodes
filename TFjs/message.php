<?php


// $length = $_POST['length'];
$resultarray = $_POST['resultarray'];
$str = $_POST['str'];

// echo $resultarray;
$filenamestr = split(' ', $str);

// print_r($filenamestr);
// echo (count($filenamestr).'-'.count($resultarray));
for($i = 0; $i < count($filenamestr); $i++)
{
	
	for($j = 0; $j < count($resultarray); $j++)
	{

		$NewString = split('---', $resultarray[$j]);
		echo $i.'----'.$NewString[1].PHP_EOL;
		if($NewString[1] == $i)
		{
			// echo $NewString[1].'-'.$i.PHP_EOL;
			$newfname = "./result/".$filenamestr[$i].".txt";
			$fp = fopen ( $newfname , 'a+' );
			fwrite($fp, $NewString[0].PHP_EOL);
		}
		
	}

}

echo '1';
?>
