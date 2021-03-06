
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let s1 = List.length l1 in
  let s2 = List.length l2 in
  if s1 < s2
  then (((clone 0 (s2 - s1)) @ l1), l2)
  else if s2 < s1 then (l1, ((clone 0 (s1 - s2)) @ l2)) else (l1, l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h != 0 then h :: t else removeZero t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = match x with | ([],[]) -> [] in
    let base = ([], []) in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let s1 = List.length l1 in
  let s2 = List.length l2 in
  if s1 < s2
  then (((clone 0 (s2 - s1)) @ l1), l2)
  else if s2 < s1 then (l1, ((clone 0 (s1 - s2)) @ l2)) else (l1, l2);;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h != 0 then h :: t else removeZero t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match snd a with
      | [] ->
          (((fst x) + (snd x)),
            [((fst x) + (snd x)) / 10; ((fst x) + (snd x)) mod 10])
      | h::t -> (0, []) in
    let base = (0, []) in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(16,16)-(16,44)
(16,22)-(16,23)
(16,42)-(16,44)
(17,16)-(17,18)
*)

(* type error slice
(16,4)-(19,51)
(16,10)-(16,44)
(16,12)-(16,44)
(16,16)-(16,44)
(16,42)-(16,44)
(17,4)-(19,51)
(17,15)-(17,23)
(19,18)-(19,32)
(19,18)-(19,44)
(19,33)-(19,34)
(19,35)-(19,39)
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
(4,12)-(9,69)
(4,15)-(9,69)
(5,2)-(9,69)
(5,11)-(5,25)
(5,11)-(5,22)
(5,23)-(5,25)
(6,2)-(9,69)
(6,11)-(6,25)
(6,11)-(6,22)
(6,23)-(6,25)
(7,2)-(9,69)
(7,5)-(7,12)
(7,5)-(7,7)
(7,10)-(7,12)
(8,7)-(8,39)
(8,8)-(8,34)
(8,29)-(8,30)
(8,9)-(8,28)
(8,10)-(8,15)
(8,16)-(8,17)
(8,18)-(8,27)
(8,19)-(8,21)
(8,24)-(8,26)
(8,31)-(8,33)
(8,36)-(8,38)
(9,7)-(9,69)
(9,10)-(9,17)
(9,10)-(9,12)
(9,15)-(9,17)
(9,23)-(9,55)
(9,24)-(9,26)
(9,28)-(9,54)
(9,49)-(9,50)
(9,29)-(9,48)
(9,30)-(9,35)
(9,36)-(9,37)
(9,38)-(9,47)
(9,39)-(9,41)
(9,44)-(9,46)
(9,51)-(9,53)
(9,61)-(9,69)
(9,62)-(9,64)
(9,66)-(9,68)
(11,19)-(12,75)
(12,2)-(12,75)
(12,8)-(12,9)
(12,23)-(12,25)
(12,36)-(12,75)
(12,39)-(12,45)
(12,39)-(12,40)
(12,44)-(12,45)
(12,51)-(12,57)
(12,51)-(12,52)
(12,56)-(12,57)
(12,63)-(12,75)
(12,63)-(12,73)
(12,74)-(12,75)
(14,11)-(20,34)
(14,14)-(20,34)
(15,2)-(20,34)
(15,11)-(19,51)
(16,4)-(19,51)
(16,10)-(16,44)
(16,12)-(16,44)
(16,16)-(16,44)
(16,22)-(16,23)
(16,42)-(16,44)
(17,4)-(19,51)
(17,15)-(17,23)
(17,16)-(17,18)
(17,20)-(17,22)
(18,4)-(19,51)
(18,15)-(18,55)
(18,15)-(18,27)
(18,28)-(18,41)
(18,29)-(18,37)
(18,38)-(18,40)
(18,42)-(18,55)
(18,43)-(18,51)
(18,52)-(18,54)
(19,4)-(19,51)
(19,18)-(19,44)
(19,18)-(19,32)
(19,33)-(19,34)
(19,35)-(19,39)
(19,40)-(19,44)
(19,48)-(19,51)
(20,2)-(20,34)
(20,2)-(20,12)
(20,13)-(20,34)
(20,14)-(20,17)
(20,18)-(20,33)
(20,19)-(20,26)
(20,27)-(20,29)
(20,30)-(20,32)
*)
