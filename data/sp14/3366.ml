
let sqsum xs =
  let f a x = a + (x * x) in let base = 0 in List.fold_left f base xs;;

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      sqsum
        ((mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @ [h * i]);;


(* fix

let sqsum xs =
  let f a x = a + (x * x) in let base = 0 in List.fold_left f base xs;;

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      [sqsum
         ((mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @ [h * i])];;

*)

(* changed spans
(9,6)-(10,77)
*)

(* type error slice
(2,3)-(3,71)
(2,10)-(3,69)
(3,2)-(3,69)
(3,8)-(3,25)
(3,14)-(3,15)
(3,14)-(3,25)
(3,29)-(3,69)
(3,45)-(3,59)
(3,45)-(3,69)
(3,60)-(3,61)
(6,2)-(10,77)
(7,10)-(7,12)
(9,6)-(9,11)
(9,6)-(10,77)
*)

(* all spans
(2,10)-(3,69)
(3,2)-(3,69)
(3,8)-(3,25)
(3,10)-(3,25)
(3,14)-(3,25)
(3,14)-(3,15)
(3,18)-(3,25)
(3,19)-(3,20)
(3,23)-(3,24)
(3,29)-(3,69)
(3,40)-(3,41)
(3,45)-(3,69)
(3,45)-(3,59)
(3,60)-(3,61)
(3,62)-(3,66)
(3,67)-(3,69)
(5,19)-(10,77)
(5,21)-(10,77)
(6,2)-(10,77)
(6,8)-(6,18)
(6,8)-(6,16)
(6,17)-(6,18)
(7,10)-(7,12)
(9,6)-(10,77)
(9,6)-(9,11)
(10,8)-(10,77)
(10,67)-(10,68)
(10,9)-(10,66)
(10,10)-(10,20)
(10,21)-(10,22)
(10,23)-(10,65)
(10,24)-(10,32)
(10,33)-(10,64)
(10,34)-(10,42)
(10,43)-(10,61)
(10,54)-(10,60)
(10,54)-(10,55)
(10,58)-(10,60)
(10,62)-(10,63)
(10,69)-(10,76)
(10,70)-(10,75)
(10,70)-(10,71)
(10,74)-(10,75)
*)
