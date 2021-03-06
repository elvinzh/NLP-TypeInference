
let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ (sep ^ t) in
      let base = "" in let l = sl in List.fold_left f base l;;


(* fix

let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ (sep ^ x) in
      let base = "" in let l = sl in List.fold_left f base l;;

*)

(* changed spans
(6,29)-(6,30)
*)

(* type error slice
(3,2)-(7,60)
(6,22)-(6,31)
(6,27)-(6,28)
(6,29)-(6,30)
*)

(* all spans
(2,18)-(7,60)
(2,22)-(7,60)
(3,2)-(7,60)
(3,8)-(3,10)
(4,10)-(4,12)
(6,6)-(7,60)
(6,12)-(6,31)
(6,14)-(6,31)
(6,18)-(6,31)
(6,20)-(6,21)
(6,18)-(6,19)
(6,22)-(6,31)
(6,27)-(6,28)
(6,23)-(6,26)
(6,29)-(6,30)
(7,6)-(7,60)
(7,17)-(7,19)
(7,23)-(7,60)
(7,31)-(7,33)
(7,37)-(7,60)
(7,37)-(7,51)
(7,52)-(7,53)
(7,54)-(7,58)
(7,59)-(7,60)
*)
