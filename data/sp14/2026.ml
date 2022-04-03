
let rec clone x n =
  match n > 0 with | false  -> [] | true  -> [x] @ ((clone x n) - 1);;


(* fix

let rec clone x n =
  match n > 0 with | false  -> [] | true  -> x :: (clone x (n - 1));;

*)

(* changed spans
(3,45)-(3,48)
(3,45)-(3,68)
(3,49)-(3,50)
(3,51)-(3,68)
(3,61)-(3,62)
*)

(* type error slice
(2,3)-(3,70)
(2,14)-(3,68)
(2,16)-(3,68)
(3,2)-(3,68)
(3,31)-(3,33)
(3,45)-(3,68)
(3,49)-(3,50)
(3,51)-(3,68)
(3,52)-(3,63)
(3,53)-(3,58)
*)

(* all spans
(2,14)-(3,68)
(2,16)-(3,68)
(3,2)-(3,68)
(3,8)-(3,13)
(3,8)-(3,9)
(3,12)-(3,13)
(3,31)-(3,33)
(3,45)-(3,68)
(3,49)-(3,50)
(3,45)-(3,48)
(3,46)-(3,47)
(3,51)-(3,68)
(3,52)-(3,63)
(3,53)-(3,58)
(3,59)-(3,60)
(3,61)-(3,62)
(3,66)-(3,67)
*)
