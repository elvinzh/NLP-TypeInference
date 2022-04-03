
let sqsum xs =
  let f a x = match x with | [] -> 0 | x::xs' -> x * x in
  let base = List.hd xs in List.fold_left f base xs;;


(* fix

let sqsum xs =
  let f a x = (a * a) + (x * x) in
  let base = List.hd xs in List.fold_left f base xs;;

*)

(* changed spans
(3,14)-(3,54)
(3,20)-(3,21)
(3,35)-(3,36)
(3,49)-(3,50)
*)

(* type error slice
(3,2)-(4,51)
(3,8)-(3,54)
(3,10)-(3,54)
(3,14)-(3,54)
(3,20)-(3,21)
(3,35)-(3,36)
(4,2)-(4,51)
(4,13)-(4,20)
(4,13)-(4,23)
(4,21)-(4,23)
(4,27)-(4,41)
(4,27)-(4,51)
(4,42)-(4,43)
(4,44)-(4,48)
(4,49)-(4,51)
*)

(* all spans
(2,10)-(4,51)
(3,2)-(4,51)
(3,8)-(3,54)
(3,10)-(3,54)
(3,14)-(3,54)
(3,20)-(3,21)
(3,35)-(3,36)
(3,49)-(3,54)
(3,49)-(3,50)
(3,53)-(3,54)
(4,2)-(4,51)
(4,13)-(4,23)
(4,13)-(4,20)
(4,21)-(4,23)
(4,27)-(4,51)
(4,27)-(4,41)
(4,42)-(4,43)
(4,44)-(4,48)
(4,49)-(4,51)
*)
