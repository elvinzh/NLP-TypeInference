
let pipe fs =
  let f a x g = a (x g) in
  let base = match fs with | (b,c)::t -> f b c | [] -> (fun x  -> x) in
  List.fold_left f base fs;;


(* fix

let pipe fs =
  let f a x g = a (x g) in
  let base = match fs with | h::t -> f h h | [] -> (fun x  -> x) in
  List.fold_left f base fs;;

*)

(* changed spans
(4,13)-(4,68)
(4,43)-(4,44)
(4,45)-(4,46)
(4,55)-(4,68)
*)

(* type error slice
(4,13)-(4,68)
(4,19)-(4,21)
(4,41)-(4,42)
(4,41)-(4,46)
(4,45)-(4,46)
(5,2)-(5,16)
(5,2)-(5,26)
(5,17)-(5,18)
(5,24)-(5,26)
*)

(* all spans
(2,9)-(5,26)
(3,2)-(5,26)
(3,8)-(3,23)
(3,10)-(3,23)
(3,12)-(3,23)
(3,16)-(3,23)
(3,16)-(3,17)
(3,18)-(3,23)
(3,19)-(3,20)
(3,21)-(3,22)
(4,2)-(5,26)
(4,13)-(4,68)
(4,19)-(4,21)
(4,41)-(4,46)
(4,41)-(4,42)
(4,43)-(4,44)
(4,45)-(4,46)
(4,55)-(4,68)
(4,66)-(4,67)
(5,2)-(5,26)
(5,2)-(5,16)
(5,17)-(5,18)
(5,19)-(5,23)
(5,24)-(5,26)
*)
