
let rec rmzhelp l =
  match l with | [] -> [] | x::xs' -> if x = 0 then rmzhelp xs' else x :: xs';;

let rec foldr f b x n = if n > 0 then f x (foldr f b x (n - 1)) else b;;

let rec clone x n = foldr (fun y  -> fun m  -> y :: m) [] x n;;

let padZero l1 l2 =
  if (List.length l1) > (List.length l2)
  then (clone 0 ((List.length l1) - (List.length l2))) @ l2
  else (clone 0 ((List.length l2) - (List.length l1))) @ l1;;

let rec removeZero l =
  match l with | [] -> [] | x::xs' -> if x = 0 then rmzhelp xs' else x :: xs';;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      let (c,d) = x in
      match a with
      | (n,listSum) ->
          (match listSum with
           | [] ->
               if ((n + c) + d) < 10
               then (0, [n; (n + c) + d])
               else ((n + 1), [n + 1; ((n + c) + d) mod 10])
           | h::t ->
               if ((n + c) + d) < 10
               then (0, ([0; (c + d) + h] @ t))
               else
                 [((n + 1), (((h + c) + d) / 10));
                 (((h + c) + d) mod 10)
                 ::
                 t]) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;


(* fix

let rec rmzhelp l =
  match l with | [] -> [] | x::xs' -> if x = 0 then rmzhelp xs' else x :: xs';;

let rec foldr f b x n = if n > 0 then f x (foldr f b x (n - 1)) else b;;

let rec clone x n = foldr (fun y  -> fun m  -> y :: m) [] x n;;

let padZero l1 l2 =
  if (List.length l1) > (List.length l2)
  then (l1, ((clone 0 ((List.length l1) - (List.length l2))) @ l2))
  else (((clone 0 ((List.length l2) - (List.length l1))) @ l1), l2);;

let rec removeZero l =
  match l with | [] -> [] | x::xs' -> if x = 0 then rmzhelp xs' else x :: xs';;

let bigAdd l1 l2 =
  let add (l1,l2) =
    let f a x =
      match x with
      | (c,d) ->
          (match a with
           | (n,listSum) ->
               (match listSum with
                | [] ->
                    if ((n + c) + d) < 10
                    then (0, ([n] @ [(n + c) + d]))
                    else ((n + 1), ([n + 1] @ [((n + c) + d) mod 10]))
                | h::t ->
                    if ((n + c) + d) < 10
                    then (0, ([0] @ ([(c + d) + h] @ t)))
                    else
                      ((n + 1),
                        ([((h + c) + d) / 10] @ ([((h + c) + d) mod 10] @ t))))) in
    let base = (0, []) in
    let args = List.rev (List.combine l1 l2) in
    let (_,res) = List.fold_left f base args in res in
  removeZero (add (padZero l1 l2));;

*)

(* changed spans
(11,7)-(11,59)
(12,7)-(12,59)
(14,19)-(15,77)
(20,6)-(35,20)
(26,24)-(26,40)
(26,25)-(26,26)
(26,28)-(26,39)
(27,30)-(27,59)
(27,31)-(27,36)
(27,38)-(27,58)
(30,25)-(30,41)
(30,29)-(30,40)
(32,17)-(35,19)
(32,28)-(32,48)
(33,17)-(33,39)
(33,17)-(35,18)
*)

(* type error slice
(9,3)-(12,61)
(9,12)-(12,59)
(9,15)-(12,59)
(10,2)-(12,59)
(12,7)-(12,59)
(12,55)-(12,56)
(18,2)-(39,34)
(18,11)-(38,51)
(29,15)-(35,19)
(30,20)-(30,47)
(32,17)-(35,19)
(32,18)-(32,49)
(33,17)-(35,18)
(39,13)-(39,34)
(39,14)-(39,17)
(39,18)-(39,33)
(39,19)-(39,26)
*)

(* all spans
(2,16)-(3,77)
(3,2)-(3,77)
(3,8)-(3,9)
(3,23)-(3,25)
(3,38)-(3,77)
(3,41)-(3,46)
(3,41)-(3,42)
(3,45)-(3,46)
(3,52)-(3,63)
(3,52)-(3,59)
(3,60)-(3,63)
(3,69)-(3,77)
(3,69)-(3,70)
(3,74)-(3,77)
(5,14)-(5,70)
(5,16)-(5,70)
(5,18)-(5,70)
(5,20)-(5,70)
(5,24)-(5,70)
(5,27)-(5,32)
(5,27)-(5,28)
(5,31)-(5,32)
(5,38)-(5,63)
(5,38)-(5,39)
(5,40)-(5,41)
(5,42)-(5,63)
(5,43)-(5,48)
(5,49)-(5,50)
(5,51)-(5,52)
(5,53)-(5,54)
(5,55)-(5,62)
(5,56)-(5,57)
(5,60)-(5,61)
(5,69)-(5,70)
(7,14)-(7,61)
(7,16)-(7,61)
(7,20)-(7,61)
(7,20)-(7,25)
(7,26)-(7,54)
(7,37)-(7,53)
(7,47)-(7,53)
(7,47)-(7,48)
(7,52)-(7,53)
(7,55)-(7,57)
(7,58)-(7,59)
(7,60)-(7,61)
(9,12)-(12,59)
(9,15)-(12,59)
(10,2)-(12,59)
(10,5)-(10,40)
(10,5)-(10,21)
(10,6)-(10,17)
(10,18)-(10,20)
(10,24)-(10,40)
(10,25)-(10,36)
(10,37)-(10,39)
(11,7)-(11,59)
(11,55)-(11,56)
(11,7)-(11,54)
(11,8)-(11,13)
(11,14)-(11,15)
(11,16)-(11,53)
(11,17)-(11,33)
(11,18)-(11,29)
(11,30)-(11,32)
(11,36)-(11,52)
(11,37)-(11,48)
(11,49)-(11,51)
(11,57)-(11,59)
(12,7)-(12,59)
(12,55)-(12,56)
(12,7)-(12,54)
(12,8)-(12,13)
(12,14)-(12,15)
(12,16)-(12,53)
(12,17)-(12,33)
(12,18)-(12,29)
(12,30)-(12,32)
(12,36)-(12,52)
(12,37)-(12,48)
(12,49)-(12,51)
(12,57)-(12,59)
(14,19)-(15,77)
(15,2)-(15,77)
(15,8)-(15,9)
(15,23)-(15,25)
(15,38)-(15,77)
(15,41)-(15,46)
(15,41)-(15,42)
(15,45)-(15,46)
(15,52)-(15,63)
(15,52)-(15,59)
(15,60)-(15,63)
(15,69)-(15,77)
(15,69)-(15,70)
(15,74)-(15,77)
(17,11)-(39,34)
(17,14)-(39,34)
(18,2)-(39,34)
(18,11)-(38,51)
(19,4)-(38,51)
(19,10)-(35,20)
(19,12)-(35,20)
(20,6)-(35,20)
(20,18)-(20,19)
(21,6)-(35,20)
(21,12)-(21,13)
(23,10)-(35,20)
(23,17)-(23,24)
(25,15)-(27,60)
(25,18)-(25,36)
(25,18)-(25,31)
(25,19)-(25,26)
(25,20)-(25,21)
(25,24)-(25,25)
(25,29)-(25,30)
(25,34)-(25,36)
(26,20)-(26,41)
(26,21)-(26,22)
(26,24)-(26,40)
(26,25)-(26,26)
(26,28)-(26,39)
(26,28)-(26,35)
(26,29)-(26,30)
(26,33)-(26,34)
(26,38)-(26,39)
(27,20)-(27,60)
(27,21)-(27,28)
(27,22)-(27,23)
(27,26)-(27,27)
(27,30)-(27,59)
(27,31)-(27,36)
(27,31)-(27,32)
(27,35)-(27,36)
(27,38)-(27,58)
(27,38)-(27,51)
(27,39)-(27,46)
(27,40)-(27,41)
(27,44)-(27,45)
(27,49)-(27,50)
(27,56)-(27,58)
(29,15)-(35,19)
(29,18)-(29,36)
(29,18)-(29,31)
(29,19)-(29,26)
(29,20)-(29,21)
(29,24)-(29,25)
(29,29)-(29,30)
(29,34)-(29,36)
(30,20)-(30,47)
(30,21)-(30,22)
(30,24)-(30,46)
(30,42)-(30,43)
(30,25)-(30,41)
(30,26)-(30,27)
(30,29)-(30,40)
(30,29)-(30,36)
(30,30)-(30,31)
(30,34)-(30,35)
(30,39)-(30,40)
(30,44)-(30,45)
(32,17)-(35,19)
(32,18)-(32,49)
(32,19)-(32,26)
(32,20)-(32,21)
(32,24)-(32,25)
(32,28)-(32,48)
(32,29)-(32,42)
(32,30)-(32,37)
(32,31)-(32,32)
(32,35)-(32,36)
(32,40)-(32,41)
(32,45)-(32,47)
(33,17)-(35,18)
(33,17)-(33,39)
(33,18)-(33,31)
(33,19)-(33,26)
(33,20)-(33,21)
(33,24)-(33,25)
(33,29)-(33,30)
(33,36)-(33,38)
(35,17)-(35,18)
(36,4)-(38,51)
(36,15)-(36,22)
(36,16)-(36,17)
(36,19)-(36,21)
(37,4)-(38,51)
(37,15)-(37,44)
(37,15)-(37,23)
(37,24)-(37,44)
(37,25)-(37,37)
(37,38)-(37,40)
(37,41)-(37,43)
(38,4)-(38,51)
(38,18)-(38,44)
(38,18)-(38,32)
(38,33)-(38,34)
(38,35)-(38,39)
(38,40)-(38,44)
(38,48)-(38,51)
(39,2)-(39,34)
(39,2)-(39,12)
(39,13)-(39,34)
(39,14)-(39,17)
(39,18)-(39,33)
(39,19)-(39,26)
(39,27)-(39,29)
(39,30)-(39,32)
*)
