import { StaticImageData } from "next/image";

export interface Card {
  title: string;
  description: string;
  src: StaticImageData | string;
  width: number;
  height: number;
  heightClass: string;
}

export interface MasonryProps {
  cards: Card[];
}
