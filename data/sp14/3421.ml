
let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ (sep ^ x) in
      let base = h in let l = t in List.fold_left f base l;;

let stringOfList f l = "[" ^ ((sepConcat "; " (List.map f "" l)) ^ "]");;


(* fix

let rec sepConcat sep sl =
  match sl with
  | [] -> ""
  | h::t ->
      let f a x = a ^ (sep ^ x) in
      let base = h in let l = t in List.fold_left f base l;;

let stringOfList f l = "[" ^ ((sepConcat "; " (List.map f l)) ^ "]");;

*)

(* changed spans
(9,46)-(9,63)
(9,58)-(9,60)
*)

(* type error slice
(9,46)-(9,63)
(9,47)-(9,55)
*)

(* all spans
(2,18)-(7,58)
(2,22)-(7,58)
(3,2)-(7,58)
(3,8)-(3,10)
(4,10)-(4,12)
(6,6)-(7,58)
(6,12)-(6,31)
(6,14)-(6,31)
(6,18)-(6,31)
(6,20)-(6,21)
(6,18)-(6,19)
(6,22)-(6,31)
(6,27)-(6,28)
(6,23)-(6,26)
(6,29)-(6,30)
(7,6)-(7,58)
(7,17)-(7,18)
(7,22)-(7,58)
(7,30)-(7,31)
(7,35)-(7,58)
(7,35)-(7,49)
(7,50)-(7,51)
(7,52)-(7,56)
(7,57)-(7,58)
(9,17)-(9,71)
(9,19)-(9,71)
(9,23)-(9,71)
(9,27)-(9,28)
(9,23)-(9,26)
(9,29)-(9,71)
(9,65)-(9,66)
(9,30)-(9,64)
(9,31)-(9,40)
(9,41)-(9,45)
(9,46)-(9,63)
(9,47)-(9,55)
(9,56)-(9,57)
(9,58)-(9,60)
(9,61)-(9,62)
(9,67)-(9,70)
*)
