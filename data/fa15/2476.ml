
let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) = (List.length l2)
  then (l1, l2)
  else
    if (List.length l1) < (List.length l2)
    then (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2)
    else (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if not (h = 0) then l else removeZero t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (b,c) ->
          (match a with
           | (carry,sum) ->
               (match sum with
                | [] ->
                    if ((carry + b) + c) < 10
                    then (0, [carry; (carry + b) + c])
                    else
                      ((carry + 1), ((carry + 1) ::
                        (((carry + b) + c) mod 10)))
                | h::t ->
                    if ((b + c) + h) < 10
                    then (0, ([0] @ ([(b + c) + h] @ t)))
                    else
                      ((carry + 1),
                        (((((h + b) + c) / 10) :: (((h + b) + c) mod 10)) @ t)))) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec clone x n = if n <= 0 then [] else x :: (clone x (n - 1));;

let padZero l1 l2 =
  if (List.length l1) = (List.length l2)
  then (l1, l2)
  else
    if (List.length l1) < (List.length l2)
    then (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2)
    else (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if not (h = 0) then l else removeZero t;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (b,c) ->
          (match a with
           | (carry,sum) ->
               (match sum with
                | [] ->
                    if ((carry + b) + c) < 10
                    then (0, [carry; (carry + b) + c])
                    else ((carry + 1), [carry + 1; ((carry + b) + c) mod 10])
                | h::t ->
                    if ((b + c) + h) < 10
                    then (0, ([0] @ ([(b + c) + h] @ t)))
                    else
                      ((carry + 1),
                        ([((h + b) + c) / 10] @ ([((h + b) + c) mod 10] @ t))))) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(27,36)-(28,51)
(34,25)-(34,73)
(34,50)-(34,72)
*)

(* type error slice
(27,36)-(28,51)
(28,24)-(28,50)
(34,25)-(34,73)
(34,50)-(34,72)
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
(4,12)-(10,69)
(4,15)-(10,69)
(5,2)-(10,69)
(5,5)-(5,40)
(5,5)-(5,21)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,15)
(6,8)-(6,10)
(6,12)-(6,14)
(8,4)-(10,69)
(8,7)-(8,42)
(8,7)-(8,23)
(8,8)-(8,19)
(8,20)-(8,22)
(8,26)-(8,42)
(8,27)-(8,38)
(8,39)-(8,41)
(9,9)-(9,69)
(9,10)-(9,64)
(9,59)-(9,60)
(9,11)-(9,58)
(9,12)-(9,17)
(9,18)-(9,19)
(9,20)-(9,57)
(9,21)-(9,37)
(9,22)-(9,33)
(9,34)-(9,36)
(9,40)-(9,56)
(9,41)-(9,52)
(9,53)-(9,55)
(9,61)-(9,63)
(9,66)-(9,68)
(10,9)-(10,69)
(10,10)-(10,12)
(10,14)-(10,68)
(10,63)-(10,64)
(10,15)-(10,62)
(10,16)-(10,21)
(10,22)-(10,23)
(10,24)-(10,61)
(10,25)-(10,41)
(10,26)-(10,37)
(10,38)-(10,40)
(10,44)-(10,60)
(10,45)-(10,56)
(10,57)-(10,59)
(10,65)-(10,67)
(12,19)-(13,75)
(13,2)-(13,75)
(13,8)-(13,9)
(13,23)-(13,25)
(13,36)-(13,75)
(13,39)-(13,50)
(13,39)-(13,42)
(13,43)-(13,50)
(13,44)-(13,45)
(13,48)-(13,49)
(13,56)-(13,57)
(13,63)-(13,75)
(13,63)-(13,73)
(13,74)-(13,75)
(15,11)-(38,34)
(15,14)-(38,34)
(16,2)-(38,34)
(16,11)-(37,51)
(17,4)-(37,51)
(17,10)-(34,81)
(17,12)-(34,81)
(18,6)-(34,81)
(18,12)-(18,13)
(20,10)-(34,81)
(20,17)-(20,18)
(22,15)-(34,80)
(22,22)-(22,25)
(24,20)-(28,52)
(24,23)-(24,45)
(24,23)-(24,40)
(24,24)-(24,35)
(24,25)-(24,30)
(24,33)-(24,34)
(24,38)-(24,39)
(24,43)-(24,45)
(25,25)-(25,54)
(25,26)-(25,27)
(25,29)-(25,53)
(25,30)-(25,35)
(25,37)-(25,52)
(25,37)-(25,48)
(25,38)-(25,43)
(25,46)-(25,47)
(25,51)-(25,52)
(27,22)-(28,52)
(27,23)-(27,34)
(27,24)-(27,29)
(27,32)-(27,33)
(27,36)-(28,51)
(27,37)-(27,48)
(27,38)-(27,43)
(27,46)-(27,47)
(28,24)-(28,50)
(28,25)-(28,42)
(28,26)-(28,37)
(28,27)-(28,32)
(28,35)-(28,36)
(28,40)-(28,41)
(28,47)-(28,49)
(30,20)-(34,79)
(30,23)-(30,41)
(30,23)-(30,36)
(30,24)-(30,31)
(30,25)-(30,26)
(30,29)-(30,30)
(30,34)-(30,35)
(30,39)-(30,41)
(31,25)-(31,57)
(31,26)-(31,27)
(31,29)-(31,56)
(31,34)-(31,35)
(31,30)-(31,33)
(31,31)-(31,32)
(31,36)-(31,55)
(31,51)-(31,52)
(31,37)-(31,50)
(31,38)-(31,49)
(31,38)-(31,45)
(31,39)-(31,40)
(31,43)-(31,44)
(31,48)-(31,49)
(31,53)-(31,54)
(33,22)-(34,79)
(33,23)-(33,34)
(33,24)-(33,29)
(33,32)-(33,33)
(34,24)-(34,78)
(34,74)-(34,75)
(34,25)-(34,73)
(34,26)-(34,46)
(34,27)-(34,40)
(34,28)-(34,35)
(34,29)-(34,30)
(34,33)-(34,34)
(34,38)-(34,39)
(34,43)-(34,45)
(34,50)-(34,72)
(34,51)-(34,64)
(34,52)-(34,59)
(34,53)-(34,54)
(34,57)-(34,58)
(34,62)-(34,63)
(34,69)-(34,71)
(34,76)-(34,77)
(35,4)-(37,51)
(35,15)-(35,22)
(35,16)-(35,17)
(35,19)-(35,21)
(36,4)-(37,51)
(36,15)-(36,44)
(36,15)-(36,23)
(36,24)-(36,44)
(36,25)-(36,37)
(36,38)-(36,40)
(36,41)-(36,43)
(37,4)-(37,51)
(37,18)-(37,44)
(37,18)-(37,32)
(37,33)-(37,34)
(37,35)-(37,39)
(37,40)-(37,44)
(37,48)-(37,51)
(38,2)-(38,34)
(38,2)-(38,12)
(38,13)-(38,34)
(38,14)-(38,17)
(38,18)-(38,33)
(38,19)-(38,26)
(38,27)-(38,29)
(38,30)-(38,32)
*)
