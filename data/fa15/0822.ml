
let rec listReverse l = match l with | [] -> [] | h::t -> [h; listReverse t];;


(* fix

let rec listReverse l =
  match l with | [] -> [] | h::t -> h :: (listReverse t);;

*)

(* changed spans
(2,58)-(2,76)
*)

(* type error slice
(2,3)-(2,78)
(2,20)-(2,76)
(2,24)-(2,76)
(2,58)-(2,76)
(2,62)-(2,73)
(2,62)-(2,75)
*)

(* all spans
(2,20)-(2,76)
(2,24)-(2,76)
(2,30)-(2,31)
(2,45)-(2,47)
(2,58)-(2,76)
(2,59)-(2,60)
(2,62)-(2,75)
(2,62)-(2,73)
(2,74)-(2,75)
*)