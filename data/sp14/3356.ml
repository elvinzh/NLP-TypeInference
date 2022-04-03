
let rec clone x n = if n > 0 then x :: (clone x (n - 1)) else [];;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1), l2)
  else (l1, (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (v1,v2) ->
          (match a with
           | (list1,list2) ->
               (match list1 with
                | [] ->
                    ((((v1 + v2) / 10) :: list1), (((v1 + v2) mod 10) ::
                      list2))
                | h::t ->
                    (((((v1 + v2) + h) / 10) :: list1),
                      ((((v1 + v2) + h) mod 10) :: list2)))) in
    let base = ([], []) in
    let args = List.append (List.rev (List.combine l1 l2)) [(0, 0)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      let rec helper acc v =
        if v = 0 then acc else helper ((v mod 10) :: acc) (v / 10) in
      let rec adder x = match x with | [] -> [] | h::t -> bigAdd h (adder t) in
      adder
        ((mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @
           [helper [] (h * i)]);;


(* fix

let rec clone x n = if n > 0 then x :: (clone x (n - 1)) else [];;

let padZero l1 l2 =
  if (List.length l1) < (List.length l2)
  then ((List.append (clone 0 ((List.length l2) - (List.length l1))) l1), l2)
  else (l1, (List.append (clone 0 ((List.length l1) - (List.length l2))) l2));;

let rec removeZero l =
  match l with | [] -> [] | h::t -> if h = 0 then removeZero t else l;;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (v1,v2) ->
          (match a with
           | (list1,list2) ->
               (match list1 with
                | [] ->
                    ((((v1 + v2) / 10) :: list1), (((v1 + v2) mod 10) ::
                      list2))
                | h::t ->
                    (((((v1 + v2) + h) / 10) :: list1),
                      ((((v1 + v2) + h) mod 10) :: list2)))) in
    let base = ([], []) in
    let args = List.append (List.rev (List.combine l1 l2)) [(0, 0)] in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      let rec helper acc v =
        if v = 0 then acc else helper ((v mod 10) :: acc) (v / 10) in
      let rec adder x = match x with | [] -> [] | h::t -> bigAdd h (adder t) in
      (mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @
        (helper [] (h * i));;

*)

(* changed spans
(38,6)-(38,11)
(38,6)-(40,31)
(40,11)-(40,30)
*)

(* type error slice
(4,3)-(7,79)
(4,12)-(7,77)
(4,15)-(7,77)
(6,8)-(6,72)
(6,9)-(6,20)
(6,21)-(6,68)
(6,22)-(6,27)
(6,69)-(6,71)
(7,12)-(7,76)
(7,13)-(7,24)
(7,25)-(7,72)
(7,26)-(7,31)
(7,73)-(7,75)
(12,3)-(29,36)
(12,11)-(29,34)
(12,14)-(29,34)
(29,18)-(29,33)
(29,19)-(29,26)
(29,27)-(29,29)
(29,30)-(29,32)
(31,3)-(40,33)
(31,19)-(40,31)
(31,21)-(40,31)
(32,2)-(40,31)
(35,6)-(40,31)
(37,6)-(40,31)
(37,24)-(37,76)
(37,58)-(37,64)
(37,58)-(37,76)
(37,65)-(37,66)
(37,67)-(37,76)
(37,68)-(37,73)
(37,74)-(37,75)
(38,6)-(38,11)
(38,6)-(40,31)
(39,8)-(40,31)
(39,9)-(39,66)
(39,10)-(39,20)
(39,67)-(39,68)
*)

(* all spans
(2,14)-(2,64)
(2,16)-(2,64)
(2,20)-(2,64)
(2,23)-(2,28)
(2,23)-(2,24)
(2,27)-(2,28)
(2,34)-(2,56)
(2,34)-(2,35)
(2,39)-(2,56)
(2,40)-(2,45)
(2,46)-(2,47)
(2,48)-(2,55)
(2,49)-(2,50)
(2,53)-(2,54)
(2,62)-(2,64)
(4,12)-(7,77)
(4,15)-(7,77)
(5,2)-(7,77)
(5,5)-(5,40)
(5,5)-(5,21)
(5,6)-(5,17)
(5,18)-(5,20)
(5,24)-(5,40)
(5,25)-(5,36)
(5,37)-(5,39)
(6,7)-(6,77)
(6,8)-(6,72)
(6,9)-(6,20)
(6,21)-(6,68)
(6,22)-(6,27)
(6,28)-(6,29)
(6,30)-(6,67)
(6,31)-(6,47)
(6,32)-(6,43)
(6,44)-(6,46)
(6,50)-(6,66)
(6,51)-(6,62)
(6,63)-(6,65)
(6,69)-(6,71)
(6,74)-(6,76)
(7,7)-(7,77)
(7,8)-(7,10)
(7,12)-(7,76)
(7,13)-(7,24)
(7,25)-(7,72)
(7,26)-(7,31)
(7,32)-(7,33)
(7,34)-(7,71)
(7,35)-(7,51)
(7,36)-(7,47)
(7,48)-(7,50)
(7,54)-(7,70)
(7,55)-(7,66)
(7,67)-(7,69)
(7,73)-(7,75)
(9,19)-(10,69)
(10,2)-(10,69)
(10,8)-(10,9)
(10,23)-(10,25)
(10,36)-(10,69)
(10,39)-(10,44)
(10,39)-(10,40)
(10,43)-(10,44)
(10,50)-(10,62)
(10,50)-(10,60)
(10,61)-(10,62)
(10,68)-(10,69)
(12,11)-(29,34)
(12,14)-(29,34)
(13,2)-(29,34)
(13,11)-(28,51)
(14,4)-(28,51)
(14,10)-(25,60)
(14,12)-(25,60)
(15,6)-(25,60)
(15,12)-(15,13)
(17,10)-(25,60)
(17,17)-(17,18)
(19,15)-(25,59)
(19,22)-(19,27)
(21,20)-(22,29)
(21,21)-(21,48)
(21,22)-(21,38)
(21,23)-(21,32)
(21,24)-(21,26)
(21,29)-(21,31)
(21,35)-(21,37)
(21,42)-(21,47)
(21,50)-(22,28)
(21,51)-(21,69)
(21,52)-(21,61)
(21,53)-(21,55)
(21,58)-(21,60)
(21,66)-(21,68)
(22,22)-(22,27)
(24,20)-(25,58)
(24,21)-(24,54)
(24,22)-(24,44)
(24,23)-(24,38)
(24,24)-(24,33)
(24,25)-(24,27)
(24,30)-(24,32)
(24,36)-(24,37)
(24,41)-(24,43)
(24,48)-(24,53)
(25,22)-(25,57)
(25,23)-(25,47)
(25,24)-(25,39)
(25,25)-(25,34)
(25,26)-(25,28)
(25,31)-(25,33)
(25,37)-(25,38)
(25,44)-(25,46)
(25,51)-(25,56)
(26,4)-(28,51)
(26,15)-(26,23)
(26,16)-(26,18)
(26,20)-(26,22)
(27,4)-(28,51)
(27,15)-(27,67)
(27,15)-(27,26)
(27,27)-(27,58)
(27,28)-(27,36)
(27,37)-(27,57)
(27,38)-(27,50)
(27,51)-(27,53)
(27,54)-(27,56)
(27,59)-(27,67)
(27,60)-(27,66)
(27,61)-(27,62)
(27,64)-(27,65)
(28,4)-(28,51)
(28,18)-(28,44)
(28,18)-(28,32)
(28,33)-(28,34)
(28,35)-(28,39)
(28,40)-(28,44)
(28,48)-(28,51)
(29,2)-(29,34)
(29,2)-(29,12)
(29,13)-(29,34)
(29,14)-(29,17)
(29,18)-(29,33)
(29,19)-(29,26)
(29,27)-(29,29)
(29,30)-(29,32)
(31,19)-(40,31)
(31,21)-(40,31)
(32,2)-(40,31)
(32,8)-(32,18)
(32,8)-(32,16)
(32,17)-(32,18)
(33,10)-(33,12)
(35,6)-(40,31)
(35,21)-(36,66)
(35,25)-(36,66)
(36,8)-(36,66)
(36,11)-(36,16)
(36,11)-(36,12)
(36,15)-(36,16)
(36,22)-(36,25)
(36,31)-(36,66)
(36,31)-(36,37)
(36,38)-(36,57)
(36,39)-(36,49)
(36,40)-(36,41)
(36,46)-(36,48)
(36,53)-(36,56)
(36,58)-(36,66)
(36,59)-(36,60)
(36,63)-(36,65)
(37,6)-(40,31)
(37,20)-(37,76)
(37,24)-(37,76)
(37,30)-(37,31)
(37,45)-(37,47)
(37,58)-(37,76)
(37,58)-(37,64)
(37,65)-(37,66)
(37,67)-(37,76)
(37,68)-(37,73)
(37,74)-(37,75)
(38,6)-(40,31)
(38,6)-(38,11)
(39,8)-(40,31)
(39,67)-(39,68)
(39,9)-(39,66)
(39,10)-(39,20)
(39,21)-(39,22)
(39,23)-(39,65)
(39,24)-(39,32)
(39,33)-(39,64)
(39,34)-(39,42)
(39,43)-(39,61)
(39,54)-(39,60)
(39,54)-(39,55)
(39,58)-(39,60)
(39,62)-(39,63)
(40,11)-(40,30)
(40,12)-(40,29)
(40,12)-(40,18)
(40,19)-(40,21)
(40,22)-(40,29)
(40,23)-(40,24)
(40,27)-(40,28)
*)
