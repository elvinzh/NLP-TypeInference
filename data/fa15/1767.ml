
let helper f b = if (f b) = b then true else false;;

let rec wwhile (f,b) =
  let (b',c') = f b in if c' = true then wwhile (f, b') else b';;

let fixpoint (f,b) = wwhile ((helper f b), b);;


(* fix

let helper f b = if (f b) = b then (true, b) else (false, (f b));;

let rec wwhile (f,b) =
  let (b',c') = f b in if c' = true then wwhile (f, b') else b';;

let fixpoint (f,b) = wwhile ((helper f), b);;

*)

(* changed spans
(2,35)-(2,39)
(2,45)-(2,50)
(4,16)-(5,63)
(7,29)-(7,41)
(7,43)-(7,44)
*)

(* type error slice
(2,3)-(2,52)
(2,11)-(2,50)
(2,13)-(2,50)
(2,17)-(2,50)
(2,45)-(2,50)
(5,16)-(5,17)
(5,16)-(5,19)
(5,41)-(5,47)
(5,41)-(5,55)
(5,48)-(5,55)
(5,49)-(5,50)
(7,21)-(7,27)
(7,21)-(7,45)
(7,28)-(7,45)
(7,29)-(7,41)
(7,30)-(7,36)
*)

(* all spans
(2,11)-(2,50)
(2,13)-(2,50)
(2,17)-(2,50)
(2,20)-(2,29)
(2,20)-(2,25)
(2,21)-(2,22)
(2,23)-(2,24)
(2,28)-(2,29)
(2,35)-(2,39)
(2,45)-(2,50)
(4,16)-(5,63)
(5,2)-(5,63)
(5,16)-(5,19)
(5,16)-(5,17)
(5,18)-(5,19)
(5,23)-(5,63)
(5,26)-(5,35)
(5,26)-(5,28)
(5,31)-(5,35)
(5,41)-(5,55)
(5,41)-(5,47)
(5,48)-(5,55)
(5,49)-(5,50)
(5,52)-(5,54)
(5,61)-(5,63)
(7,14)-(7,45)
(7,21)-(7,45)
(7,21)-(7,27)
(7,28)-(7,45)
(7,29)-(7,41)
(7,30)-(7,36)
(7,37)-(7,38)
(7,39)-(7,40)
(7,43)-(7,44)
*)
