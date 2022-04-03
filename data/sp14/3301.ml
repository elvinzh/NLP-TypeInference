
let bigMul l1 l2 =
  let f a x = ([a], [x]) in
  let base = ([], [0]) in
  let args =
    let rec helper acc l1 l2 =
      match l1 with
      | [] -> acc
      | h::t -> helper ((h, l2) :: acc) (List.map (fun x  -> x * 10) t) l2 in
    helper [] (List.rev l1) l2 in
  let (_,res) = List.fold_left f base args in res;;


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
      bigAdd (mulByDigit i (List.rev (List.map (fun x  -> x * 10) t)))
        (helper [] (h * i));;

let bigMul l1 l2 =
  let f a x =
    match x with
    | (k,v) -> (match a with | (c,d) -> (c, (bigAdd d (mulByDigit k v)))) in
  let base = ([], [0]) in
  let args =
    let rec helper acc l1 l2 =
      match l1 with
      | [] -> acc
      | h::t -> helper ((h, l2) :: acc) (List.map (fun x  -> x * 10) t) l2 in
    helper [] (List.rev l1) l2 in
  let (_,res) = List.fold_left f base args in res;;

*)

(* changed spans
(2,11)-(11,49)
(3,2)-(11,49)
(3,14)-(3,24)
(3,15)-(3,18)
(3,16)-(3,17)
(3,20)-(3,23)
(3,21)-(3,22)
(4,2)-(11,49)
*)

(* type error slice
(3,2)-(11,49)
(3,8)-(3,24)
(3,10)-(3,24)
(3,14)-(3,24)
(3,15)-(3,18)
(3,16)-(3,17)
(11,16)-(11,30)
(11,16)-(11,42)
(11,31)-(11,32)
*)

(* all spans
(2,11)-(11,49)
(2,14)-(11,49)
(3,2)-(11,49)
(3,8)-(3,24)
(3,10)-(3,24)
(3,14)-(3,24)
(3,15)-(3,18)
(3,16)-(3,17)
(3,20)-(3,23)
(3,21)-(3,22)
(4,2)-(11,49)
(4,13)-(4,22)
(4,14)-(4,16)
(4,18)-(4,21)
(4,19)-(4,20)
(5,2)-(11,49)
(6,4)-(10,30)
(6,19)-(9,74)
(6,23)-(9,74)
(6,26)-(9,74)
(7,6)-(9,74)
(7,12)-(7,14)
(8,14)-(8,17)
(9,16)-(9,74)
(9,16)-(9,22)
(9,23)-(9,39)
(9,24)-(9,31)
(9,25)-(9,26)
(9,28)-(9,30)
(9,35)-(9,38)
(9,40)-(9,71)
(9,41)-(9,49)
(9,50)-(9,68)
(9,61)-(9,67)
(9,61)-(9,62)
(9,65)-(9,67)
(9,69)-(9,70)
(9,72)-(9,74)
(10,4)-(10,30)
(10,4)-(10,10)
(10,11)-(10,13)
(10,14)-(10,27)
(10,15)-(10,23)
(10,24)-(10,26)
(10,28)-(10,30)
(11,2)-(11,49)
(11,16)-(11,42)
(11,16)-(11,30)
(11,31)-(11,32)
(11,33)-(11,37)
(11,38)-(11,42)
(11,46)-(11,49)
*)
