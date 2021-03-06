
let rec wwhile (f,b) =
  let (value,result) = f b in if not result then value else wwhile (f, value);;

let fixpoint (f,b) =
  wwhile ((let func output = ((f output), ((f output) = b)) in func b), b);;


(* fix

let rec wwhile (f,b) =
  let (value,result) = f b in if not result then value else wwhile (f, value);;

let fixpoint (f,b) =
  wwhile ((let func input = ((f input), ((f input) = b)) in func), b);;

*)

(* changed spans
(6,20)-(6,59)
(6,33)-(6,39)
(6,46)-(6,52)
(6,63)-(6,69)
(6,72)-(6,73)
*)

(* type error slice
(3,23)-(3,24)
(3,23)-(3,26)
(3,60)-(3,66)
(3,60)-(3,77)
(3,67)-(3,77)
(3,68)-(3,69)
(6,2)-(6,8)
(6,2)-(6,74)
(6,9)-(6,74)
(6,10)-(6,70)
(6,20)-(6,59)
(6,29)-(6,59)
(6,63)-(6,67)
(6,63)-(6,69)
*)

(* all spans
(2,16)-(3,77)
(3,2)-(3,77)
(3,23)-(3,26)
(3,23)-(3,24)
(3,25)-(3,26)
(3,30)-(3,77)
(3,33)-(3,43)
(3,33)-(3,36)
(3,37)-(3,43)
(3,49)-(3,54)
(3,60)-(3,77)
(3,60)-(3,66)
(3,67)-(3,77)
(3,68)-(3,69)
(3,71)-(3,76)
(5,14)-(6,74)
(6,2)-(6,74)
(6,2)-(6,8)
(6,9)-(6,74)
(6,10)-(6,70)
(6,20)-(6,59)
(6,29)-(6,59)
(6,30)-(6,40)
(6,31)-(6,32)
(6,33)-(6,39)
(6,42)-(6,58)
(6,43)-(6,53)
(6,44)-(6,45)
(6,46)-(6,52)
(6,56)-(6,57)
(6,63)-(6,69)
(6,63)-(6,67)
(6,68)-(6,69)
(6,72)-(6,73)
*)
