
let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = (sepConcat sep a) @ x in
      let base = [] in let l = t in List.fold_left f base l;;


(* fix

let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ sep in
      let base = h in let l = t in List.fold_left f base l;;

*)

(* changed spans
(6,18)-(6,35)
(6,19)-(6,28)
(6,29)-(6,32)
(6,36)-(6,37)
(6,38)-(6,39)
(7,17)-(7,19)
*)

(* type error slice
(2,3)-(7,61)
(2,18)-(7,59)
(2,22)-(7,59)
(3,2)-(7,59)
(4,10)-(4,12)
(6,6)-(7,59)
(6,12)-(6,39)
(6,14)-(6,39)
(6,18)-(6,35)
(6,18)-(6,39)
(6,19)-(6,28)
(6,36)-(6,37)
(7,6)-(7,59)
(7,23)-(7,59)
(7,36)-(7,50)
(7,36)-(7,59)
(7,51)-(7,52)
*)

(* all spans
(2,18)-(7,59)
(2,22)-(7,59)
(3,2)-(7,59)
(3,8)-(3,10)
(4,10)-(4,12)
(6,6)-(7,59)
(6,12)-(6,39)
(6,14)-(6,39)
(6,18)-(6,39)
(6,36)-(6,37)
(6,18)-(6,35)
(6,19)-(6,28)
(6,29)-(6,32)
(6,33)-(6,34)
(6,38)-(6,39)
(7,6)-(7,59)
(7,17)-(7,19)
(7,23)-(7,59)
(7,31)-(7,32)
(7,36)-(7,59)
(7,36)-(7,50)
(7,51)-(7,52)
(7,53)-(7,57)
(7,58)-(7,59)
*)
