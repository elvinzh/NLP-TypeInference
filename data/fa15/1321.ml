
let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = x ^ a in
      let base = "" in let l = f in List.fold_left f base l;;


(* fix

let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = x ^ a in
      let base = "" in let l = t in List.fold_left f base l;;

*)

(* changed spans
(7,31)-(7,32)
*)

(* type error slice
(6,6)-(7,59)
(6,12)-(6,23)
(7,23)-(7,59)
(7,31)-(7,32)
(7,36)-(7,50)
(7,36)-(7,59)
(7,58)-(7,59)
*)

(* all spans
(2,18)-(7,59)
(2,22)-(7,59)
(3,2)-(7,59)
(3,8)-(3,10)
(4,10)-(4,12)
(6,6)-(7,59)
(6,12)-(6,23)
(6,14)-(6,23)
(6,18)-(6,23)
(6,20)-(6,21)
(6,18)-(6,19)
(6,22)-(6,23)
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
