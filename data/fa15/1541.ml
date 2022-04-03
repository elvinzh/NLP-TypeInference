
let rec wwhile (f,b) =
  match f b with | (h1,h2) -> if h2 then wwhile (f, h1) else h1;;

let fixpoint (f,b) = wwhile ((let f' b = (f b) = b in f'), b);;


(* fix

let rec wwhile (f,b) =
  match f b with | (h1,h2) -> if h2 then wwhile (f, h1) else h1;;

let fixpoint (f,b) =
  wwhile
    ((let f' b = if (f b) = b then (b, true) else ((f b), false) in f'), b);;

*)

(* changed spans
(5,41)-(5,50)
(5,54)-(5,56)
*)

(* type error slice
(3,2)-(3,63)
(3,8)-(3,9)
(3,8)-(3,11)
(3,41)-(3,47)
(3,41)-(3,55)
(3,48)-(3,55)
(3,49)-(3,50)
(5,21)-(5,27)
(5,21)-(5,61)
(5,28)-(5,61)
(5,29)-(5,57)
(5,37)-(5,50)
(5,41)-(5,50)
(5,54)-(5,56)
*)

(* all spans
(2,16)-(3,63)
(3,2)-(3,63)
(3,8)-(3,11)
(3,8)-(3,9)
(3,10)-(3,11)
(3,30)-(3,63)
(3,33)-(3,35)
(3,41)-(3,55)
(3,41)-(3,47)
(3,48)-(3,55)
(3,49)-(3,50)
(3,52)-(3,54)
(3,61)-(3,63)
(5,14)-(5,61)
(5,21)-(5,61)
(5,21)-(5,27)
(5,28)-(5,61)
(5,29)-(5,57)
(5,37)-(5,50)
(5,41)-(5,50)
(5,41)-(5,46)
(5,42)-(5,43)
(5,44)-(5,45)
(5,49)-(5,50)
(5,54)-(5,56)
(5,59)-(5,60)
*)
