
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  if len1 > len2
  then (l1, ((clone 0 (len1 - len2)) @ l2))
  else (((clone 0 (len2 - len1)) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t when h == 0 -> removeZero t | h::t -> h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with | (x,y) -> (((x + y) / 10), (((x + y) mod 10) :: a)) in
    let base = [] in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let len1 = List.length l1 in
  let len2 = List.length l2 in
  if len1 > len2
  then (l1, ((clone 0 (len1 - len2)) @ l2))
  else (((clone 0 (len2 - len1)) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t when h == 0 -> removeZero t | h::t -> h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match a with
      | (w,z) ->
          (match x with
           | (x,y) -> ((((x + y) + w) / 10), ((((x + y) + w) mod 10) :: z))) in
    let base = (0, []) in
    let args = List.combine l1 l2 in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(16,4)-(20,51)
(17,30)-(17,71)
(17,32)-(17,39)
(17,47)-(17,70)
(17,49)-(17,56)
(17,61)-(17,63)
(17,68)-(17,69)
*)

(* type error slice
(16,4)-(20,51)
(16,10)-(17,71)
(16,12)-(17,71)
(17,6)-(17,71)
(17,30)-(17,71)
(17,47)-(17,70)
(17,68)-(17,69)
(20,18)-(20,32)
(20,18)-(20,44)
(20,33)-(20,34)
*)

(* all spans
(2,14)-(2,65)
(2,16)-(2,65)
(2,20)-(2,65)
(2,23)-(2,29)
(2,23)-(2,24)
(2,28)-(2,29)
(2,35)-(2,37)
(2,43)-(2,65)
(2,43)-(2,44)
(2,48)-(2,65)
(2,49)-(2,54)
(2,55)-(2,56)
(2,57)-(2,64)
(2,58)-(2,59)
(2,62)-(2,63)
(4,12)-(9,43)
(4,15)-(9,43)
(5,2)-(9,43)
(5,13)-(5,27)
(5,13)-(5,24)
(5,25)-(5,27)
(6,2)-(9,43)
(6,13)-(6,27)
(6,13)-(6,24)
(6,25)-(6,27)
(7,2)-(9,43)
(7,5)-(7,16)
(7,5)-(7,9)
(7,12)-(7,16)
(8,7)-(8,43)
(8,8)-(8,10)
(8,12)-(8,42)
(8,37)-(8,38)
(8,13)-(8,36)
(8,14)-(8,19)
(8,20)-(8,21)
(8,22)-(8,35)
(8,23)-(8,27)
(8,30)-(8,34)
(8,39)-(8,41)
(9,7)-(9,43)
(9,8)-(9,38)
(9,33)-(9,34)
(9,9)-(9,32)
(9,10)-(9,15)
(9,16)-(9,17)
(9,18)-(9,31)
(9,19)-(9,23)
(9,26)-(9,30)
(9,35)-(9,37)
(9,40)-(9,42)
(11,19)-(12,77)
(12,2)-(12,77)
(12,8)-(12,9)
(12,38)-(12,44)
(12,38)-(12,39)
(12,43)-(12,44)
(12,23)-(12,25)
(12,48)-(12,60)
(12,48)-(12,58)
(12,59)-(12,60)
(12,71)-(12,77)
(12,71)-(12,72)
(12,76)-(12,77)
(14,11)-(21,34)
(14,14)-(21,34)
(15,2)-(21,34)
(15,11)-(20,51)
(16,4)-(20,51)
(16,10)-(17,71)
(16,12)-(17,71)
(17,6)-(17,71)
(17,12)-(17,13)
(17,30)-(17,71)
(17,31)-(17,45)
(17,32)-(17,39)
(17,33)-(17,34)
(17,37)-(17,38)
(17,42)-(17,44)
(17,47)-(17,70)
(17,48)-(17,64)
(17,49)-(17,56)
(17,50)-(17,51)
(17,54)-(17,55)
(17,61)-(17,63)
(17,68)-(17,69)
(18,4)-(20,51)
(18,15)-(18,17)
(19,4)-(20,51)
(19,15)-(19,33)
(19,15)-(19,27)
(19,28)-(19,30)
(19,31)-(19,33)
(20,4)-(20,51)
(20,18)-(20,44)
(20,18)-(20,32)
(20,33)-(20,34)
(20,35)-(20,39)
(20,40)-(20,44)
(20,48)-(20,51)
(21,2)-(21,34)
(21,2)-(21,12)
(21,13)-(21,34)
(21,14)-(21,17)
(21,18)-(21,33)
(21,19)-(21,26)
(21,27)-(21,29)
(21,30)-(21,32)
*)
