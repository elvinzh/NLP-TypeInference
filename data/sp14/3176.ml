
let pipe fs = let f a x = () in let base x = x in List.fold_left f base fs;;


(* fix

let pipe fs = let f a x = x in let base x = x in List.fold_left f base fs;;

*)

(* changed spans
(2,26)-(2,28)
*)

(* type error slice
(2,14)-(2,74)
(2,20)-(2,28)
(2,22)-(2,28)
(2,26)-(2,28)
(2,32)-(2,74)
(2,41)-(2,46)
(2,50)-(2,64)
(2,50)-(2,74)
(2,65)-(2,66)
(2,67)-(2,71)
*)

(* all spans
(2,9)-(2,74)
(2,14)-(2,74)
(2,20)-(2,28)
(2,22)-(2,28)
(2,26)-(2,28)
(2,32)-(2,74)
(2,41)-(2,46)
(2,45)-(2,46)
(2,50)-(2,74)
(2,50)-(2,64)
(2,65)-(2,66)
(2,67)-(2,71)
(2,72)-(2,74)
*)
