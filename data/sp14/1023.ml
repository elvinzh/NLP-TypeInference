
let pipe fs =
  let f a x p' = a in let base a' = a' in List.fold_left f base fs;;


(* fix

let pipe fs = let f a x = a in let base p = p in List.fold_left f base fs;;

*)

(* changed spans
(3,12)-(3,18)
(3,31)-(3,38)
(3,36)-(3,38)
(3,42)-(3,66)
*)

(* type error slice
(3,2)-(3,66)
(3,8)-(3,18)
(3,10)-(3,18)
(3,12)-(3,18)
(3,17)-(3,18)
(3,42)-(3,56)
(3,42)-(3,66)
(3,57)-(3,58)
*)

(* all spans
(2,9)-(3,66)
(3,2)-(3,66)
(3,8)-(3,18)
(3,10)-(3,18)
(3,12)-(3,18)
(3,17)-(3,18)
(3,22)-(3,66)
(3,31)-(3,38)
(3,36)-(3,38)
(3,42)-(3,66)
(3,42)-(3,56)
(3,57)-(3,58)
(3,59)-(3,63)
(3,64)-(3,66)
*)
