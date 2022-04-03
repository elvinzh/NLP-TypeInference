
let pipe fs =
  let f a x = function | f' -> f' x a in
  let base = function | y -> y in List.fold_left f base fs;;


(* fix

let pipe fs =
  let f a x = function | v -> x (a v) in
  let base = function | y -> y in List.fold_left f base fs;;

*)

(* changed spans
(3,14)-(3,37)
(3,31)-(3,33)
(3,31)-(3,37)
(3,36)-(3,37)
(4,2)-(4,58)
*)

(* type error slice
(3,2)-(4,58)
(3,8)-(3,37)
(3,10)-(3,37)
(3,14)-(3,37)
(3,31)-(3,33)
(3,31)-(3,37)
(3,36)-(3,37)
(4,34)-(4,48)
(4,34)-(4,58)
(4,49)-(4,50)
*)

(* all spans
(2,9)-(4,58)
(3,2)-(4,58)
(3,8)-(3,37)
(3,10)-(3,37)
(3,14)-(3,37)
(3,31)-(3,37)
(3,31)-(3,33)
(3,34)-(3,35)
(3,36)-(3,37)
(4,2)-(4,58)
(4,13)-(4,30)
(4,29)-(4,30)
(4,34)-(4,58)
(4,34)-(4,48)
(4,49)-(4,50)
(4,51)-(4,55)
(4,56)-(4,58)
*)
