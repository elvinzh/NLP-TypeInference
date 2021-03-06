
let sqsum xs =
  let f a x a x = a + (x * x) in
  let base = "string" in List.fold_left f base xs;;


(* fix

let sqsum xs =
  let f a x a x = a + x in
  let base = 0 in List.fold_left (fun a  -> fun x  -> a + (x * x)) 0 xs;;

*)

(* changed spans
(3,22)-(3,29)
(3,27)-(3,28)
(4,13)-(4,21)
(4,40)-(4,41)
(4,42)-(4,46)
(4,47)-(4,49)
*)

(* type error slice
(3,2)-(4,49)
(3,8)-(3,29)
(3,10)-(3,29)
(3,12)-(3,29)
(4,2)-(4,49)
(4,13)-(4,21)
(4,25)-(4,39)
(4,25)-(4,49)
(4,40)-(4,41)
(4,42)-(4,46)
*)

(* all spans
(2,10)-(4,49)
(3,2)-(4,49)
(3,8)-(3,29)
(3,10)-(3,29)
(3,12)-(3,29)
(3,14)-(3,29)
(3,18)-(3,29)
(3,18)-(3,19)
(3,22)-(3,29)
(3,23)-(3,24)
(3,27)-(3,28)
(4,2)-(4,49)
(4,13)-(4,21)
(4,25)-(4,49)
(4,25)-(4,39)
(4,40)-(4,41)
(4,42)-(4,46)
(4,47)-(4,49)
*)
