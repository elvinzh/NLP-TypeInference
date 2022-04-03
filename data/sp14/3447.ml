
let rec digitsOfInt n =
  if n <= 0
  then []
  else List.rev ((n mod 10) :: (List.rev (digitsOfInt (n / 10))));;

let rec sumList xs =
  match xs with | [] -> 0 | h::t -> h + (sumList t) | _ -> (-1);;

let rec additivePersistence n =
  let count = [0] in
  if (sumList (digitsOfInt n)) > 9 then 1 :: count else sumList count;;


(* fix

let rec digitsOfInt n =
  if n <= 0
  then []
  else List.rev ((n mod 10) :: (List.rev (digitsOfInt (n / 10))));;

let rec sumList xs =
  match xs with | [] -> 0 | h::t -> h + (sumList t) | _ -> (-1);;

let rec additivePersistence n =
  let x = 1 in
  x + 1;
  if (sumList (digitsOfInt n)) > 9
  then additivePersistence (sumList (digitsOfInt n))
  else x;;

*)

(* changed spans
(11,2)-(12,69)
(11,14)-(11,17)
(11,15)-(11,16)
(12,2)-(12,69)
(12,40)-(12,41)
(12,40)-(12,50)
(12,45)-(12,50)
(12,56)-(12,63)
(12,64)-(12,69)
*)

(* type error slice
(8,36)-(8,51)
(8,40)-(8,51)
(8,41)-(8,48)
(12,2)-(12,69)
(12,40)-(12,50)
(12,56)-(12,63)
(12,56)-(12,69)
*)

(* all spans
(2,20)-(5,65)
(3,2)-(5,65)
(3,5)-(3,11)
(3,5)-(3,6)
(3,10)-(3,11)
(4,7)-(4,9)
(5,7)-(5,65)
(5,7)-(5,15)
(5,16)-(5,65)
(5,17)-(5,27)
(5,18)-(5,19)
(5,24)-(5,26)
(5,31)-(5,64)
(5,32)-(5,40)
(5,41)-(5,63)
(5,42)-(5,53)
(5,54)-(5,62)
(5,55)-(5,56)
(5,59)-(5,61)
(7,16)-(8,63)
(8,2)-(8,63)
(8,8)-(8,10)
(8,24)-(8,25)
(8,36)-(8,51)
(8,36)-(8,37)
(8,40)-(8,51)
(8,41)-(8,48)
(8,49)-(8,50)
(8,59)-(8,63)
(10,28)-(12,69)
(11,2)-(12,69)
(11,14)-(11,17)
(11,15)-(11,16)
(12,2)-(12,69)
(12,5)-(12,34)
(12,5)-(12,30)
(12,6)-(12,13)
(12,14)-(12,29)
(12,15)-(12,26)
(12,27)-(12,28)
(12,33)-(12,34)
(12,40)-(12,50)
(12,40)-(12,41)
(12,45)-(12,50)
(12,56)-(12,69)
(12,56)-(12,63)
(12,64)-(12,69)
*)
