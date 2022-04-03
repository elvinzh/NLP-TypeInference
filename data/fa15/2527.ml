
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let sizDif = (List.length l1) - (List.length l2) in
  if sizDif > 0
  then let pad = clone 0 sizDif in (l1, (pad @ l2))
  else (let pad = clone 0 (- sizDif) in ((pad @ l1), l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h == 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x = a + x in
    let base = 0 in
    let args = List.combine (padZero (l1 l2)) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  let sizDif = (List.length l1) - (List.length l2) in
  if sizDif > 0
  then let pad = clone 0 sizDif in (l1, (pad @ l2))
  else (let pad = clone 0 (- sizDif) in ((pad @ l1), l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h == 0 then removeZero t else h :: t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (x1,x2) = x in
      let (a1,a2) = a in
      if (x1 + x2) > 10
      then (1, ((((x1 + x2) + a1) - 10) :: a2))
      else (0, (((x1 + x2) + a1) :: a2)) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(15,16)-(15,17)
(15,16)-(15,21)
(16,4)-(18,51)
(16,15)-(16,16)
(17,4)-(18,51)
(17,15)-(17,27)
(17,28)-(17,45)
(17,29)-(17,36)
(17,37)-(17,44)
*)

(* type error slice
(4,3)-(8,59)
(4,12)-(8,57)
(4,15)-(8,57)
(15,4)-(18,51)
(15,10)-(15,21)
(15,16)-(15,17)
(15,16)-(15,21)
(17,15)-(17,27)
(17,15)-(17,45)
(17,28)-(17,45)
(17,29)-(17,36)
(18,4)-(18,51)
(18,18)-(18,32)
(18,18)-(18,44)
(18,33)-(18,34)
(19,18)-(19,33)
(19,19)-(19,26)
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
(4,12)-(8,57)
(4,15)-(8,57)
(5,2)-(8,57)
(5,15)-(5,50)
(5,15)-(5,31)
(5,16)-(5,27)
(5,28)-(5,30)
(5,34)-(5,50)
(5,35)-(5,46)
(5,47)-(5,49)
(6,2)-(8,57)
(6,5)-(6,15)
(6,5)-(6,11)
(6,14)-(6,15)
(7,7)-(7,51)
(7,17)-(7,31)
(7,17)-(7,22)
(7,23)-(7,24)
(7,25)-(7,31)
(7,35)-(7,51)
(7,36)-(7,38)
(7,40)-(7,50)
(7,45)-(7,46)
(7,41)-(7,44)
(7,47)-(7,49)
(8,7)-(8,57)
(8,18)-(8,36)
(8,18)-(8,23)
(8,24)-(8,25)
(8,26)-(8,36)
(8,29)-(8,35)
(8,40)-(8,56)
(8,41)-(8,51)
(8,46)-(8,47)
(8,42)-(8,45)
(8,48)-(8,50)
(8,53)-(8,55)
(10,19)-(11,75)
(11,2)-(11,75)
(11,8)-(11,9)
(11,23)-(11,25)
(11,36)-(11,75)
(11,39)-(11,45)
(11,39)-(11,40)
(11,44)-(11,45)
(11,51)-(11,63)
(11,51)-(11,61)
(11,62)-(11,63)
(11,69)-(11,75)
(11,69)-(11,70)
(11,74)-(11,75)
(13,11)-(19,34)
(13,14)-(19,34)
(14,2)-(19,34)
(14,11)-(18,51)
(15,4)-(18,51)
(15,10)-(15,21)
(15,12)-(15,21)
(15,16)-(15,21)
(15,16)-(15,17)
(15,20)-(15,21)
(16,4)-(18,51)
(16,15)-(16,16)
(17,4)-(18,51)
(17,15)-(17,45)
(17,15)-(17,27)
(17,28)-(17,45)
(17,29)-(17,36)
(17,37)-(17,44)
(17,38)-(17,40)
(17,41)-(17,43)
(18,4)-(18,51)
(18,18)-(18,44)
(18,18)-(18,32)
(18,33)-(18,34)
(18,35)-(18,39)
(18,40)-(18,44)
(18,48)-(18,51)
(19,2)-(19,34)
(19,2)-(19,12)
(19,13)-(19,34)
(19,14)-(19,17)
(19,18)-(19,33)
(19,19)-(19,26)
(19,27)-(19,29)
(19,30)-(19,32)
*)
