
let rec wwhile (f,b) =
  match f b with | (i,true ) -> wwhile (f, i) | (i,false ) -> i;;

let fixpoint (f,b) =
  wwhile (if b = (f b) then (b, false) else (((f b), true), b));;


(* fix

let rec wwhile (f,b) =
  match f b with | (i,true ) -> wwhile (f, i) | (i,false ) -> i;;

let fixpoint (f,b) =
  let helper x = if b = (f b) then (b, false) else (b, true) in
  wwhile (helper, b);;

*)

(* changed spans
(6,2)-(6,8)
(6,2)-(6,63)
(6,9)-(6,63)
(6,45)-(6,58)
(6,46)-(6,51)
(6,47)-(6,48)
(6,60)-(6,61)
*)

(* type error slice
(6,9)-(6,63)
(6,28)-(6,38)
(6,29)-(6,30)
(6,32)-(6,37)
(6,44)-(6,62)
(6,45)-(6,58)
(6,60)-(6,61)
*)

(* all spans
(2,16)-(3,63)
(3,2)-(3,63)
(3,8)-(3,11)
(3,8)-(3,9)
(3,10)-(3,11)
(3,32)-(3,45)
(3,32)-(3,38)
(3,39)-(3,45)
(3,40)-(3,41)
(3,43)-(3,44)
(3,62)-(3,63)
(5,14)-(6,63)
(6,2)-(6,63)
(6,2)-(6,8)
(6,9)-(6,63)
(6,13)-(6,22)
(6,13)-(6,14)
(6,17)-(6,22)
(6,18)-(6,19)
(6,20)-(6,21)
(6,28)-(6,38)
(6,29)-(6,30)
(6,32)-(6,37)
(6,44)-(6,62)
(6,45)-(6,58)
(6,46)-(6,51)
(6,47)-(6,48)
(6,49)-(6,50)
(6,53)-(6,57)
(6,60)-(6,61)
*)
