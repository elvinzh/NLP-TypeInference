
let rec assoc (d,k,l) =
  let acc = l in
  let rec helper acc (d,k,l) =
    match acc with
    | [] -> d
    | h::t ->
        (match h with | (s,v) -> if s = k then v else helper t (d, k, l)) in
  helper (d, k, l) acc;;


(* fix

let rec assoc (d,k,l) =
  let acc = l in
  let rec helper acc (d,k,l) =
    match acc with
    | [] -> d
    | h::t ->
        (match h with | (s,v) -> if s = k then v else helper t (d, k, l)) in
  helper acc (d, k, l);;

*)

(* changed spans
(9,9)-(9,18)
(9,19)-(9,22)
*)

(* type error slice
(5,4)-(8,73)
(8,54)-(8,60)
(8,54)-(8,72)
(8,61)-(8,62)
(9,2)-(9,8)
(9,2)-(9,22)
(9,9)-(9,18)
*)

(* all spans
(2,15)-(9,22)
(3,2)-(9,22)
(3,12)-(3,13)
(4,2)-(9,22)
(4,17)-(8,73)
(4,22)-(8,73)
(5,4)-(8,73)
(5,10)-(5,13)
(6,12)-(6,13)
(8,8)-(8,73)
(8,15)-(8,16)
(8,33)-(8,72)
(8,36)-(8,41)
(8,36)-(8,37)
(8,40)-(8,41)
(8,47)-(8,48)
(8,54)-(8,72)
(8,54)-(8,60)
(8,61)-(8,62)
(8,63)-(8,72)
(8,64)-(8,65)
(8,67)-(8,68)
(8,70)-(8,71)
(9,2)-(9,22)
(9,2)-(9,8)
(9,9)-(9,18)
(9,10)-(9,11)
(9,13)-(9,14)
(9,16)-(9,17)
(9,19)-(9,22)
*)